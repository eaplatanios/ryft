use std::collections::HashSet;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use prost::Message;
use ryft_core::{
    ArrayIrType, ArrayOperation, ArrayType, CompilationCacheDomain, CompilationDomain, Dimension as CoreDimension,
    DimensionBounds, DimensionVariable, DispatchInputs, ExternalReferenceBinding, FlatProgram as CoreFlatProgram,
    Memory, MeshAxisType, Operation as CoreOperation, Parameterized, ProgramError, ReductionKind,
    ReferenceDischargeResult, ScatterMode, ScatterReductionKind, Shape, Sharding, ShardingDimension, SortDirection,
    StagedProgram, Type, TypeError, Typed, TypedProgram,
};
use ryft_pjrt::protos::{CompilationOptions, OpShardingType};
use ryft_pjrt::{LoadedExecutable, Program as PjrtProgram};

use crate::domains::Domain;
use crate::domains::execution::{ExecutionError, ExecutionPlan};
use crate::domains::options::{FeedbackDirectedProfile, InputBoundBucketing, Options};
use crate::domains::persistence::COMPILER_IDENTITY;
use crate::domains::programs::{CompiledProgram, InvocationMetadata, LoweredProgram};
use crate::experimental::lowering::{
    collect_session_constants, contains_unresolved_references, contains_unresolved_state,
};
use crate::experimental::ops::{Constant, FlatProgram, Operation};
use crate::sharding::sharding_from_op_sharding;
use crate::{Error, Value};

// TODO(eaplatanios): Review this module.

/// XLA option, executable compatibility, persistence, and analysis failures.
#[derive(Clone, Debug, thiserror::Error)]
pub enum CompilationError {
    /// A requested compilation option combination is invalid.
    #[error("invalid compilation options: {reason}")]
    InvalidOptions { reason: String },

    /// Signature sharding metadata has a different arity than the values it describes.
    #[error("invalid {kind} sharding count; expected {expected} but got {actual}")]
    ShardingCountMismatch { kind: &'static str, expected: usize, actual: usize },

    /// A valid external-reference configuration cannot be represented by this backend's executable ABI.
    #[error("unsupported XLA reference ABI: {reason}")]
    UnsupportedReferenceAbi { reason: String },

    /// A compiled executable does not preserve its declared invocation contract.
    #[error("incompatible XLA executable: {reason}")]
    IncompatibleExecutable { reason: String },

    /// Persistent executable bytes or metadata violate their versioned transport contract.
    #[error("invalid persistent XLA executable: {reason}")]
    InvalidPersistentExecutable { reason: String },

    /// Compiler analysis properties cannot be interpreted according to their declared schema.
    #[error("invalid XLA compilation analysis: {reason}")]
    InvalidAnalysis { reason: String },
}

/// Opaque retained-JIT specialization key derived by [`Domain`].
///
/// Exact dispatch preserves nominal [`DimensionVariable`] identity and shares its type slice with the effective staged
/// signature, so keying a call costs no deep type clone. Dispatch with input-bound bucketing instead uses a structural
/// physical-bound signature with dimension variables numbered by first occurrence, so calls routed to the same bucket
/// share one staged and compiled specialization. Bucketed keys exclude diagnostic names, which survive only in the
/// effective staged types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DispatchKey {
    /// Signature component of this key.
    kind: DispatchKeyKind,

    /// Runtime placements of the public inputs of an executable specialization whose inputs are placed over auto mesh
    /// axes, which their types never record (refer to [`Options::input_placements`]), or [`None`] when no input is.
    placements: Option<Arc<[Option<Sharding>]>>,
}

impl DispatchKey {
    /// Creates a [`DispatchKey`] without runtime placements.
    fn new(kind: DispatchKeyKind) -> Self {
        Self { kind, placements: None }
    }
}

/// Internal representation of [`DispatchKey`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum DispatchKeyKind {
    /// Exact runtime abstract signature used when bucketing is disabled, shared with the effective staged types.
    Exact(Arc<[ArrayIrType]>),

    /// Bounded signature using canonical variable indices when bucketing is enabled.
    Bucketed(BucketedDispatchSignature),
}

/// Bounded input signature with dimension variables numbered by first occurrence, used as the retained-dispatch key.
///
/// Dimension variables are numbered by first occurrence across the whole input signature under
/// [`DimensionVariable`] identity, and only their bounds are retained. Two bucketed signatures are therefore equal
/// exactly when they agree on every input's data type, layout, sharding, memory space, rank, per-axis static extents,
/// and per-axis variable-sharing pattern with matching bounds, regardless of how the variables are named.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BucketedDispatchSignature {
    /// `(lower, upper)` bounds of each distinct dimension variable, in first-occurrence order.
    variables: Vec<(usize, Option<usize>)>,

    /// Bucketed input array types with their shapes rewritten into variable-index patterns.
    input_types: Vec<BucketedArrayTypeKey>,
}

/// One bucketed input array type split into a shape-free carrier and a shape pattern using canonical variable indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BucketedArrayTypeKey {
    /// Array type with an empty shape, retaining data type, layout, sharding, and memory identity.
    carrier: ArrayType,

    /// Per-axis static extents or canonical variable indices, which also fix the array's rank.
    dimensions: Vec<BucketedDimensionKey>,
}

/// One static extent or canonical variable index of a [`BucketedArrayTypeKey`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum BucketedDimensionKey {
    /// Statically known extent. Bucketing rewrites every static input axis into a bounded dynamic axis, so this
    /// variant is unreachable for signatures produced today; it is retained so the normalization stays total over
    /// [`CoreDimension`](ryft_core::Dimension).
    Static(usize),

    /// Index of the axis' dimension variable in [`BucketedDispatchSignature::variables`].
    Variable(usize),
}

impl BucketedDispatchSignature {
    /// Builds a dispatch key from the already bucketed `input_types`, numbering dimension variables by first
    /// occurrence.
    fn new(input_types: &[ArrayType]) -> Self {
        let mut variables = Vec::<DimensionVariable>::new();
        let input_types = input_types
            .iter()
            .map(|array_type| BucketedArrayTypeKey {
                carrier: array_type.clone().with_shape(Shape::new(Vec::new())),
                dimensions: array_type
                    .shape()
                    .dimensions()
                    .iter()
                    .map(|dimension| match dimension {
                        CoreDimension::Static(extent) => BucketedDimensionKey::Static(*extent),
                        CoreDimension::Dynamic(variable) => {
                            let index =
                                variables.iter().position(|existing| existing == variable).unwrap_or_else(|| {
                                    variables.push(variable.clone());
                                    variables.len() - 1
                                });
                            BucketedDimensionKey::Variable(index)
                        }
                    })
                    .collect(),
            })
            .collect();
        let variables =
            variables.iter().map(|variable| (variable.bounds().lower(), variable.bounds().upper())).collect();
        Self { variables, input_types }
    }
}

/// Exact process-local compilation key for an XLA lowering.
///
/// Unlike the former call-site fingerprint, this key carries the complete StableHLO computation and every
/// execution-relevant field retained by [`LoweredProgram`]. Equality therefore means that cached compiled
/// artifacts are interchangeable, even when unrelated call sites happen to share the same input signature.
///
/// Dynamic dimensions are numbered by first occurrence, so diagnostic variable names do not affect identity.
/// Compilation options, shardings, target facts, runtime requirements, and effect metadata remain part of the key.
/// This complete compilation identity is distinct from a [`DispatchKey`], which selects a retained specialization.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompilationKey {
    /// Complete canonical compilation contract, compared and hashed without reducing it to a fingerprint.
    pub(super) canonical_bytes: Arc<[u8]>,
}

impl<'c> CompilationDomain for Domain<'c> {
    type DispatchKey = DispatchKey;
    type LoweredProgram = LoweredProgram;
    type CompiledProgram = CompiledProgram<'c>;
    type Options = Options;
    type Error = Error;

    fn dispatch_signature(
        inputs: DispatchInputs<'_, '_, Self::Value>,
        options: &Self::Options,
    ) -> Result<(Self::DispatchKey, Arc<[ArrayIrType]>, Option<Self::Options>), Self::Error> {
        let (input_types, values) = match inputs {
            DispatchInputs::Types(input_types) => (input_types, None),
            DispatchInputs::Values { values, types } => (types, Some(values)),
        };
        let input_count = input_types.len();
        // Only independent executable requests apply bucketing. Staged calls retain their caller's signature.
        let bucketing = values.as_ref().and(options.input_bound_bucketing);
        let (key, input_types) = normalize_dispatch_signature(input_types, options, bucketing)?;
        let Some(inputs) = values.filter(|_| options.input_shardings.is_none()) else {
            return Ok((key, input_types, None));
        };

        // Only placements that name auto mesh axes are missing from the input types, and the placement vector is only
        // allocated once such an input appears, so that ordinary calls pay for no allocation.
        let mut placements = None::<Vec<Option<Sharding>>>;
        for (index, input) in inputs.enumerate() {
            match input {
                Value::Array(array) if array.sharding().references_auto_axis() => {
                    placements.get_or_insert_with(|| vec![None; index]).push(Some(array.sharding().clone()))
                }
                _ => {
                    if let Some(placements) = &mut placements {
                        placements.push(None);
                    }
                }
            }
        }
        let Some(mut placements) = placements else {
            return Ok((key, input_types, None));
        };
        placements.resize(input_count, None);
        let placements: Arc<[_]> = placements.into();
        let mut options = options.clone();
        options.input_placements = Some(Arc::clone(&placements));
        Ok((DispatchKey { placements: Some(placements), ..key }, input_types, Some(options)))
    }

    fn lower(&self, staged: &StagedProgram<Self>) -> Result<LoweredProgram, Self::Error> {
        let program = staged.lifted_program()?;
        let capture_count = staged.source_program().captures().len();
        let input_types = staged.input_types();
        let output_types = staged.output_types();
        let options = staged.options();
        if apply_input_shardings(input_types.to_vec(), options)? != input_types {
            return Err(Error::Compilation(CompilationError::InvalidOptions {
                reason: "staged input types do not carry the shardings requested by `input_shardings`; apply them to \
                         the input types before staging (e.g., with `with_input_shardings`)"
                    .to_string(),
            }));
        }
        validate_reference_outputs(program.as_ref(), capture_count, output_types)?;
        validate_output_shardings(output_types, options)?;

        // Captured arrays are placed as they are at runtime, including over the auto mesh axes that their types do not
        // record, so that executions need not reshard them.
        let capture_placements = staged
            .source_program()
            .captures()
            .iter()
            .map(|capture| match capture {
                Value::Array(array) if array.sharding().references_auto_axis() => Some(array.sharding().clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let placed_options;
        let options = if capture_placements.iter().any(Option::is_some) {
            placed_options = Options { capture_placements: Some(capture_placements.into()), ..options.clone() };
            &placed_options
        } else {
            options
        };
        let lowered_program = self.lower_program(program.as_ref(), capture_count, options)?;
        let lowered_output_types = lowered_program.output_types().to_vec();
        // Lowering gives the program outputs the types of the output sharding overrides without their auto mesh axes,
        // so the staged output types are checked against the lowered ones with those same overrides applied.
        let declared_output_types =
            apply_signature_shardings(output_types.to_vec(), options.output_shardings.as_deref(), "output")?;
        validate_output_types(&declared_output_types, &lowered_output_types)?;
        Ok(lowered_program)
    }

    // The cached executable depends only on the types of the session constants of `program`, so their arrays are
    // resolved through this domain's session for every request, including cache hits (refer to `SessionConstant`).
    fn compile(&self, program: &LoweredProgram) -> Result<(Arc<CompiledProgram<'c>>, Vec<Value<'c>>), Self::Error> {
        let session_arrays = program
            .session_constants
            .iter()
            .map(|constant| Ok(Value::Array(self.session().constant(self, constant)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        Ok((self.compilation_context().compile(self, program)?, session_arrays))
    }

    fn execute(&self, program: &CompiledProgram<'c>, inputs: Vec<Value<'c>>) -> Result<Vec<Value<'c>>, Self::Error> {
        self.execute_stateless(program, inputs).map(|(outputs, _)| outputs)
    }
}

impl<'c> Domain<'c> {
    /// Lowers a staged program to StableHLO and derives its complete logical and physical invocation contract for this
    /// domain.
    pub(super) fn lower_program(
        &self,
        program: &FlatProgram,
        capture_count: usize,
        options: &Options,
    ) -> Result<LoweredProgram, Error> {
        // Kernel selection validates the actual attached body and discharges only its owned local state before the
        // ordinary reference-discharge and effect gates. Existing lowering, caching, and runtime lifecycle then apply.
        let selected = crate::kernels::select_kernels(program, options.kernel_compiler.as_ref(), || {
            let target = self
                .target()
                .map_err(|error| crate::kernels::KernelEmbeddingError::Invalid { message: error.to_string() })?;
            crate::kernels::KernelExecutionFacts::from_target(target, &options.mesh)
        })?;
        let program = selected.as_ref().unwrap_or(program);
        if capture_count > program.input_count() {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "capture_count is {capture_count}, but the program has only {} flat input(s)",
                    program.input_count(),
                ),
            }
            .into());
        }
        // Avoid cloning the dominant reference-free program path. Discharge returns such a program untouched with an
        // empty external-state list, so the same metadata can cross the lowering boundary by borrow instead.
        if !contains_unresolved_references(program) {
            return self.lower_verified_reference_free_program(
                program,
                capture_count,
                program.output_count(),
                &[],
                options,
            );
        }
        // Discharge the exact lifted program so attached-region capture references resolve against the same lexical
        // capture prefixes used during staging. Programs that actually contain references are cloned because
        // discharge consumes and rewrites them; the reference-free identity path above remains borrowed.
        let discharged = program.clone().discharge_references_in_capture_lifted_program(capture_count)?;
        self.lower_discharged_program(&discharged, options)
    }

    /// Lowers one validated reference-discharge artifact through the ordinary array-only XLA path.
    fn lower_discharged_program(
        &self,
        discharged: &ReferenceDischargeResult<Constant, Operation>,
        options: &Options,
    ) -> Result<LoweredProgram, Error> {
        // The backend verifier remains independent from core discharge and runs before boundary projection,
        // partitioner selection, or StableHLO construction. The reference scan runs first so a malformed artifact
        // containing both a reference and ordered state receives the more precise reference diagnostic.
        if contains_unresolved_references(discharged.program()) {
            return Err(ProgramError::UnsupportedOperation {
                message: "discharged program still contains references, which XLA lowering cannot represent"
                    .to_string(),
            }
            .into());
        }
        self.lower_verified_reference_free_program(
            discharged.program(),
            discharged.capture_count(),
            discharged.output_count(),
            discharged.external_reference_bindings(),
            options,
        )
    }

    /// Lowers one program after an independent verifier has established that it contains no references.
    fn lower_verified_reference_free_program(
        &self,
        program: &FlatProgram,
        capture_count: usize,
        output_count: usize,
        reference_states: &[ExternalReferenceBinding],
        options: &Options,
    ) -> Result<LoweredProgram, Error> {
        if contains_unresolved_state(program) {
            return Err(ProgramError::UnsupportedOperation {
                message: "discharged program still contains ordered state, which XLA lowering cannot represent"
                    .to_string(),
            }
            .into());
        }
        validate_data_dependent_compilation(program)?;
        if output_count > program.output_count() {
            return Err(ProgramError::MalformedProgram(format!(
                "output count {output_count} exceeds the complete discharged output count {}",
                program.output_count(),
            ))
            .into());
        }
        // Each process binds, executes, and commits only its addressable shards of a reference, so a mesh that spans
        // processes is admitted as long as this process participates in it (refer to `Target::is_participating_mesh`).
        if !reference_states.is_empty() && !self.target()?.is_participating_mesh(&options.mesh) {
            return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                reason:
                    "external reference state requires a mesh of devices that this client sees under the processes \
                         that own them, at least one of which this process addresses"
                        .to_string(),
            }));
        }
        if !reference_states.is_empty() && options.input_bound_bucketing.is_some() {
            return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                reason: "external reference inputs do not support input-bound bucketing".to_string(),
            }));
        }
        let input_types = program.input_types();
        let (capture_types, public_input_types) = input_types.split_at(capture_count);
        if let Some(donation_flags) = &options.donation_flags
            && donation_flags.len() != public_input_types.len()
        {
            return Err(Error::Execution(ExecutionError::DonationFlagCountMismatch {
                expected_count: public_input_types.len(),
                actual_count: donation_flags.len(),
            }));
        }
        let mut donation_flags =
            options.donation_flags.clone().unwrap_or_else(|| vec![false; public_input_types.len()]);

        let mut effective_capture_types = capture_types
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        // Arrays that session constants refer to enter the executable as hidden arguments that follow the captures, so
        // that the executable depends only on their types. Execution supplies them with the captures (refer to
        // `CompilationDomain::compile`).
        let session_constants = collect_session_constants(program);
        effective_capture_types.extend(session_constants.iter().map(|constant| constant.r#type().into_owned()));
        let lowered_capture_count = effective_capture_types.len();
        let effective_public_input_types = public_input_types
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let effective_input_types = effective_capture_types
            .iter()
            .cloned()
            .chain(effective_public_input_types.iter().cloned())
            .collect::<Vec<_>>();
        let reference_state_input_indices = reference_states
            .iter()
            .map(|state| state.source().flat_input_index(lowered_capture_count))
            .collect::<Result<Vec<_>, _>>()?;
        if reference_state_input_indices.windows(2).any(|indices| indices[0] >= indices[1]) {
            return Err(ProgramError::MalformedProgram(
                "external reference states must follow strictly increasing logical input order".to_string(),
            )
            .into());
        }
        if !reference_states
            .iter()
            .filter_map(ExternalReferenceBinding::output_index)
            .eq(output_count..program.output_count())
        {
            return Err(ProgramError::MalformedProgram(
                "hidden discharged outputs must be covered exactly once by mutated external state".to_string(),
            )
            .into());
        }
        for &logical_input_index in &reference_state_input_indices {
            let input_type = effective_input_types.get(logical_input_index).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "external state logical input {logical_input_index} is out of range",
                ))
            })?;
            if let Some(sharding) = input_type.sharding()
                && !sharding.mesh().has_same_topology(options.mesh.logical_mesh())
            {
                return Err(ryft_core::ShardingError::MeshMismatch {
                    expected: options.mesh.logical_mesh().clone(),
                    actual: sharding.mesh().clone(),
                }
                .into());
            }
            if input_type.data_type().is_zero() {
                return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                    reason: format!(
                        "external reference state at logical input {logical_input_index} cannot use a zero-space type \
                         because it has no executable buffer",
                    ),
                }));
            }
            if input_type.memory() != Memory::Device {
                return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                    reason: format!(
                        "external reference state at logical input {logical_input_index} must use device memory"
                    ),
                }));
            }
            if input_type
                .shape()
                .dimensions()
                .iter()
                .any(|dimension| dimension.value().is_none() && dimension.upper_bound().is_none())
            {
                return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                    reason: format!(
                        "external reference state at logical input {logical_input_index} has an unbounded dynamic \
                         dimension",
                    ),
                }));
            }
        }
        let program_output_types = program.output_types();
        let public_output_types = apply_signature_shardings(
            program_output_types[..output_count].to_vec(),
            options.output_shardings.as_deref(),
            "output",
        )?;
        // Executable arguments are placed as their runtime arrays are (or as an input override requests), including
        // over the auto mesh axes that their types never record. Session constants are placed as their retained arrays.
        let source_capture_count = lowered_capture_count - session_constants.len();
        let logical_argument_shardings = effective_input_types
            .iter()
            .enumerate()
            .map(|(index, array_type)| {
                let placement = if index < source_capture_count {
                    options
                        .capture_placements
                        .as_deref()
                        .and_then(|placements| placements.get(index).cloned().flatten())
                } else if index < lowered_capture_count {
                    let constant = &session_constants[index - source_capture_count];
                    Some(self.session().constant(self, constant)?.sharding().clone())
                } else {
                    let index = index - lowered_capture_count;
                    options.input_shardings.as_deref().and_then(|shardings| shardings.get(index).cloned()).or_else(
                        || {
                            options
                                .input_placements
                                .as_deref()
                                .and_then(|placements| placements.get(index).cloned().flatten())
                        },
                    )
                };
                Ok(placement.or_else(|| array_type.sharding().cloned()).unwrap_or_else(|| {
                    Sharding::replicated(options.mesh.logical_mesh().clone(), array_type.shape().rank())
                }))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        for (state, &logical_input_index) in reference_states.iter().zip(&reference_state_input_indices) {
            if effective_input_types[logical_input_index].static_shape().is_none() {
                if state.output_index().is_some() {
                    return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                        reason: format!(
                            "mutated bounded-dynamic external state input {logical_input_index} is unsupported because \
                             backend alias compatibility has not been verified",
                        ),
                    }));
                }
                if !logical_argument_shardings[logical_input_index].is_replicated() {
                    return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                        reason: format!(
                            "bounded-dynamic external state input {logical_input_index} must use replicated sharding",
                        ),
                    }));
                }
                // Bounded-dynamic inputs are padded through host copies, which have not been verified across processes.
                if !self.target()?.is_fully_addressable_single_process_mesh(&options.mesh) {
                    return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                        reason: format!(
                            "bounded-dynamic external state input {logical_input_index} requires a fully addressable \
                             single-process mesh",
                        ),
                    }));
                }
            }
        }
        let mut output_types = public_output_types
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        // Outputs are placed as an output override requests, including over auto mesh axes, and otherwise as their
        // types are.
        let output_placement = |index: usize, array_type: &ArrayType| {
            options
                .output_shardings
                .as_deref()
                .and_then(|shardings| shardings.get(index).cloned())
                .or_else(|| array_type.sharding().cloned())
        };
        let mut output_shardings = output_types
            .iter()
            .enumerate()
            .map(|(index, array_type)| {
                output_placement(index, array_type).unwrap_or_else(|| {
                    Sharding::replicated(options.mesh.logical_mesh().clone(), array_type.shape().rank())
                })
            })
            .collect::<Vec<_>>();
        let result_shardings = if reference_states.is_empty() {
            output_types
                .iter()
                .enumerate()
                .map(|(index, array_type)| output_placement(index, array_type))
                .collect::<Option<Vec<_>>>()
        } else {
            let mut shardings = output_shardings.clone();
            // Mutated bindings cover the hidden output suffix in order, as validated above. Derive both the hidden
            // state type and its placement from the same input binding in one pass.
            for (state, &logical_input_index) in reference_states.iter().zip(&reference_state_input_indices) {
                if state.is_mutated() {
                    output_types.push(effective_input_types[logical_input_index].clone());
                    shardings.push(logical_argument_shardings[logical_input_index].clone());
                    output_shardings.push(logical_argument_shardings[logical_input_index].clone());
                }
            }
            Some(shardings)
        };
        // Shardy rejects bounded-dynamic tensors anywhere in the module, not only at the computation boundary, so a
        // program whose boundary is static but whose interior derives a dynamic extent through a gateway operation
        // still has to take the replica path. This is why the whole region arena is inspected instead of just the
        // boundary types, which it already subsumes.
        let use_shardy_partitioner = has_only_static_array_types(program);
        // A `shard_map` body only ever has static boundary types, so it can coexist with dynamic tensors staged
        // elsewhere in the same program. Its `sdy.manual_computation` lowering is meaningless without Shardy on any
        // device count, and bounded-dynamic tensors force Shardy off for the whole module.
        if !use_shardy_partitioner && contains_shard_map(program) {
            return Err(Error::Lowering(crate::LoweringError::UnsupportedOperation {
                operation: ryft_core::SHARD_MAP_OPERATION_NAME.to_string(),
                reason: "bounded-dynamic programs cannot contain shard_map regions because sdy.manual_computation \
                         requires the Shardy partitioner"
                    .to_string(),
            }));
        }
        if !use_shardy_partitioner && options.mesh.devices().len() > 1 {
            if logical_argument_shardings
                .iter()
                .chain(output_types.iter().filter_map(ArrayType::sharding))
                .any(|sharding| !sharding.is_replicated())
            {
                return Err(Error::Compilation(CompilationError::InvalidOptions {
                    reason: "bounded-dynamic multi-device programs currently require fully replicated shardings \
                             because Shardy rejects dynamic tensors"
                        .to_string(),
                }));
            }
        }
        let compilation_options = jit_compilation_options(
            self.compilation_options.as_ref(),
            options.mesh.devices().len(),
            use_shardy_partitioner,
            options.feedback_directed_profile.as_ref(),
        );
        // The target platform gates platform-specific lowerings such as fused attention; a failed
        // platform query degrades to the portable lowerings instead of failing compilation.
        let target = self.target()?;
        let target_platform = Some(target.platform_name());
        // Shardy currently rejects bounded-dynamic tensors. Fully replicated dynamic programs need no partitioning,
        // so lower them without Shardy metadata and execute one replica per mesh device instead.
        let lowered_argument_shardings = use_shardy_partitioner.then_some(logical_argument_shardings.as_slice());
        let lowered_result_shardings = if use_shardy_partitioner { result_shardings.as_deref() } else { None };
        // Public outputs without an override leave their placement over auto mesh axes to the compiler, like JAX
        // outputs without `out_shardings`, and compilation reads that placement back from the executable. Axes of size
        // one admit only one placement, so meshes without larger auto axes keep every result closed. Outputs with
        // reduction state or manual variation, and hidden reference states, are always placed as declared, and outputs
        // with unconstrained dimensions already leave their whole placement to the compiler (refer to `ExecutionPlan`).
        let open_outputs = output_shardings
            .iter()
            .enumerate()
            .map(|(index, sharding)| {
                lowered_result_shardings.is_some()
                    && index < output_count
                    && options.output_shardings.is_none()
                    && sharding.mesh().axes().iter().any(|axis| axis.r#type() == MeshAxisType::Auto && axis.size() > 1)
                    && sharding.unreduced_axes().is_empty()
                    && sharding.reduced_axes().is_empty()
                    && sharding.varying_manual_axes().is_empty()
                    && !sharding.dimensions().contains(&ShardingDimension::Unconstrained)
            })
            .collect::<Vec<_>>();
        let lowered_module = crate::experimental::lowering::lower_mlir_module_for_program_with_reference_state(
            program,
            effective_capture_types.as_slice(),
            &effective_public_input_types,
            &output_types,
            "main",
            lowered_argument_shardings,
            lowered_result_shardings,
            target_platform.as_deref(),
            reference_states,
            options.ragged_dot_lowering_strategy,
            &open_outputs,
        )?;
        let (stable_hlo, signature, requires_assertion_handler, runtime_requirements) = lowered_module.into_parts();
        for (mapping, donation) in
            signature.input_mapping()[lowered_capture_count..].iter().zip(donation_flags.iter_mut())
        {
            if mapping.is_none() {
                *donation = false;
            }
        }
        let expected_argument_shardings = signature.physical_input_shardings(logical_argument_shardings.as_slice());

        Ok(LoweredProgram {
            stable_hlo: stable_hlo.into(),
            compilation_options,
            input_types: effective_input_types.into(),
            output_types: output_types.into(),
            output_count,
            reference_states: reference_states.into(),
            signature,
            requires_assertion_handler,
            runtime_requirements,
            donation_flags: donation_flags.into(),
            capture_count: lowered_capture_count,
            session_constants: session_constants.into(),
            expected_argument_shardings: expected_argument_shardings.into(),
            output_shardings: output_shardings.into(),
            open_outputs: open_outputs.into(),
            mesh: options.mesh.clone(),
            platform_name: target.platform_name().into(),
            platform_version: target.platform_version().into(),
            device_kinds: target.mesh_device_kinds(&options.mesh)?.into(),
            kernel_execution_facts: if program.entry_region_ref().computation_regions().any(|region| {
                region.instructions().iter().any(|instruction| match instruction.operation() {
                    Operation::CustomCall(operation) | Operation::Array(ArrayOperation::CustomCall(operation)) => {
                        operation.attributes().iter().any(|(name, _)| name == "ryft.kernel.schema")
                    }
                    _ => false,
                })
            }) {
                Some(crate::kernels::KernelExecutionFacts::from_target(target, &options.mesh)?.configuration_key()?)
            } else {
                None
            },
            compiler_identity: COMPILER_IDENTITY.as_str().into(),
            xla_flags: std::env::var("XLA_FLAGS").unwrap_or_default().into(),
        })
    }

    /// Compiles a lowering through this domain's client after resolving its runtime requirements, then derives the
    /// reusable execution plan.
    pub(crate) fn compile_program(&self, program: &LoweredProgram) -> Result<CompiledProgram<'c>, Error> {
        self.ensure_runtime_requirements(&program.runtime_requirements)?;
        let pjrt_program = PjrtProgram::Mlir { bytecode: program.stable_hlo.as_bytes().to_vec() };
        let compilation_start = Instant::now();
        let executable = self.client().compile(&pjrt_program, &program.compilation_options)?;
        let compilation_duration = compilation_start.elapsed();
        let output_shardings = resolve_output_shardings(&executable, program)?;
        let plan = ExecutionPlan::new(
            &executable,
            &program.input_types,
            &program.output_types,
            &output_shardings,
            &program.signature,
            &program.donation_flags,
            program.capture_count,
            &program.mesh,
        )?;
        Ok(CompiledProgram {
            executable: Arc::new(executable),
            input_types: Arc::clone(&program.input_types),
            output_types: Arc::clone(&program.output_types),
            output_count: program.output_count,
            reference_states: Arc::clone(&program.reference_states),
            signature: program.signature.clone(),
            runtime_requirements: Arc::clone(&program.runtime_requirements),
            requires_assertion_handler: program.requires_assertion_handler,
            donation_flags: Arc::clone(&program.donation_flags),
            capture_count: program.capture_count,
            expected_argument_shardings: Arc::clone(&program.expected_argument_shardings),
            output_shardings,
            mesh: program.mesh.clone(),
            compilation_options: program.compilation_options.clone(),
            platform_name: Arc::clone(&program.platform_name),
            platform_version: Arc::clone(&program.platform_version),
            device_kinds: Arc::clone(&program.device_kinds),
            kernel_execution_facts: program.kernel_execution_facts.clone(),
            compiler_identity: Arc::clone(&program.compiler_identity),
            xla_flags: Arc::clone(&program.xla_flags),
            compilation_duration: Some(compilation_duration),
            analysis: Arc::new(OnceLock::new()),
            plan: Arc::new(plan),
        })
    }

    /// Validates ownership and the complete invocation contract before replacing a compiled executable.
    pub(crate) fn validate_replacement(
        &self,
        current: &CompiledProgram<'c>,
        replacement: &CompiledProgram<'c>,
    ) -> Result<(), Error> {
        self.validate_program_owner(current)?;
        self.validate_program_owner(replacement)?;
        InvocationMetadata::from(current).validate_replacement(replacement.into())
    }
}

impl<'c> CompilationCacheDomain for Domain<'c> {
    type CacheKey = CompilationKey;

    #[inline]
    fn compilation_key(&self, program: &Self::LoweredProgram) -> Result<Self::CacheKey, Self::Error> {
        program.compilation_key()
    }

    #[inline]
    fn compile_uncached(&self, program: &Self::LoweredProgram) -> Result<Self::CompiledProgram, Self::Error> {
        self.compile_program(program)
    }

    fn validate_cached_program(
        &self,
        program: &Self::LoweredProgram,
        compiled: &Self::CompiledProgram,
    ) -> Result<(), Self::Error> {
        // Fresh compilations copy this metadata verbatim, so this rejects only reused programs whose invocation
        // contract does not match the requesting lowering (e.g., a corrupted or colliding persistent or exchanged
        // artifact). The stateful transaction trusts this metadata, so a mismatch must be caught before the first
        // call.
        match InvocationMetadata::from(program).incompatible_field(compiled.into()) {
            Some(field) => Err(Error::Compilation(CompilationError::IncompatibleExecutable {
                reason: format!("cached executable has incompatible {field}"),
            })),
            None => Ok(()),
        }
    }

    #[inline]
    fn persistent_cache_key(&self, key: &Self::CacheKey) -> Option<Vec<u8>> {
        Some(key.canonical_bytes.to_vec())
    }

    #[inline]
    fn serialize_program(&self, program: &Self::CompiledProgram) -> Result<Option<Vec<u8>>, Self::Error> {
        program.serialize()
    }

    #[inline]
    fn deserialize_program(&self, bytes: &[u8]) -> Result<Option<Self::CompiledProgram>, Self::Error> {
        self.deserialize_compiled_program(bytes)
    }
}

/// Applies input sharding declarations and optional executable bucketing, without inspecting runtime values.
fn normalize_dispatch_signature(
    input_types: Vec<ArrayIrType>,
    options: &Options,
    bucketing: Option<InputBoundBucketing>,
) -> Result<(DispatchKey, Arc<[ArrayIrType]>), Error> {
    let input_types = apply_input_shardings(input_types, options)?;
    let Some(bucketing) = bucketing else {
        let input_types: Arc<[ArrayIrType]> = input_types.into();
        return Ok((DispatchKey::new(DispatchKeyKind::Exact(input_types.clone())), input_types));
    };
    if options.mesh.devices().len() > 1 {
        return Err(Error::Compilation(CompilationError::InvalidOptions {
            reason: "input-bound bucketing is unsupported on multi-device meshes because bucketed dynamic \
                     signatures require disabling the Shardy partitioner"
                .to_string(),
        }));
    }
    let mut bucketed_array_types = Vec::with_capacity(input_types.len());
    for (input_index, r#type) in input_types.iter().enumerate() {
        let array_type = match r#type {
            ArrayIrType::Array(array_type) => array_type,
            ArrayIrType::Dimension(_) => {
                return Err(Error::Compilation(CompilationError::InvalidOptions {
                    reason: format!(
                        "input-bound bucketing only supports array inputs, but input {} has first-class dimension \
                         type {}",
                        input_index, r#type,
                    ),
                }));
            }
            ArrayIrType::Reference(_) => {
                return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                    reason: format!("external reference input {input_index} does not support input-bound bucketing"),
                }));
            }
        };
        let dimensions = array_type
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .map(|(axis, dimension)| match dimension {
                CoreDimension::Static(extent) => {
                    let bound = match bucketing {
                        InputBoundBucketing::PowerOfTwo => extent.checked_next_power_of_two(),
                    }
                    .and_then(|bound| bound.checked_add(1))
                    .ok_or_else(|| ProgramError::InvalidArgument {
                        message: format!(
                            "cannot select a finite input-bound bucket for input {input_index} axis {axis} with \
                             extent {extent}",
                        ),
                    })?;
                    let bounds = DimensionBounds::new(0, Some(bound))?;
                    Ok(CoreDimension::Dynamic(DimensionVariable::new(
                        format!("input_{input_index}_axis_{axis}"),
                        bounds,
                    )))
                }
                CoreDimension::Dynamic(variable) => Ok(CoreDimension::Dynamic(variable.clone())),
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        bucketed_array_types.push(array_type.clone().with_shape(Shape::new(dimensions)));
    }
    let key = BucketedDispatchSignature::new(bucketed_array_types.as_slice());
    let input_types = bucketed_array_types.into_iter().map(ArrayIrType::Array).collect::<Vec<_>>();
    Ok((DispatchKey::new(DispatchKeyKind::Bucketed(key)), input_types.into()))
}

/// Returns the runtime placement of every output of `program` compiled as `executable`. The placement of an output
/// that lowering left open over auto mesh axes (refer to [`LoweredProgram::open_outputs`]) is the placement that the
/// compiler chose for it, which must agree with the placement that the output type records over the other mesh axes.
fn resolve_output_shardings(
    executable: &LoadedExecutable<'_>,
    program: &LoweredProgram,
) -> Result<Arc<[Sharding]>, Error> {
    if !program.open_outputs.iter().any(|open| *open) {
        return Ok(Arc::clone(&program.output_shardings));
    }
    let incompatible = |reason: String| Error::Compilation(CompilationError::IncompatibleExecutable { reason });
    let op_shardings = executable.executable()?.output_shardings()?.ok_or_else(|| {
        incompatible("the PJRT plugin does not report the output shardings that the compiler chose".to_string())
    })?;
    let op_shardings = match op_shardings.as_slice() {
        [sharding] if sharding.r#type == OpShardingType::Tuple as i32 => sharding.tuple_shardings.clone(),
        _ => op_shardings,
    };
    program
        .output_shardings
        .iter()
        .zip(program.open_outputs.iter())
        .enumerate()
        .map(|(index, (declared, &open))| {
            let Some(physical_index) = program.signature.output_mapping()[index].filter(|_| open) else {
                return Ok(declared.clone());
            };
            let op_sharding = op_shardings.get(physical_index).ok_or_else(|| {
                incompatible(format!("the executable reports no sharding for output {physical_index}"))
            })?;
            sharding_from_op_sharding(op_sharding, declared.mesh(), declared.rank())
                .filter(|sharding| sharding.without_auto_axes() == declared.without_auto_axes())
                .ok_or_else(|| {
                    incompatible(format!(
                        "the compiler placed output {index} as {op_sharding:?}, which is not a placement over the mesh \
                         axes that agrees with its declared sharding `{declared}`",
                    ))
                })
        })
        .collect()
}

/// Returns `input_types` with the input sharding overrides of `options` (i.e., [`Options::input_shardings`])
/// applied, or `input_types` unchanged when `options` has none. The overrides must be applied before tracing, because
/// tracers carry their input types and sharding-aware operations read them. Retained dispatch applies them through
/// [`CompilationDomain::dispatch_signature`], the staging entry points of this crate apply them through
/// [`with_input_shardings`], and lowering rejects staged functions whose input types do not carry them. External
/// reference inputs are rejected when overrides are present, because holder state derives its effective sharding from
/// the compiled reference ABI.
fn apply_input_shardings(input_types: Vec<ArrayIrType>, options: &Options) -> Result<Vec<ArrayIrType>, Error> {
    let Some(input_shardings) = options.input_shardings.as_deref() else {
        return Ok(input_types);
    };
    if input_types.iter().any(|r#type| r#type.is_reference()) {
        return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
            reason: "external reference inputs do not support explicit input shardings".to_string(),
        }));
    }
    apply_signature_shardings(input_types, Some(input_shardings), "input")
}

/// Applies [`Options::input_shardings`] to `input_types`, preserving their parameter structure. Returns the input
/// types unchanged when no overrides are configured.
///
/// Call this before [`ryft_core::StagedFunction::stage`] when using the generic core staging API with XLA sharding
/// overrides. Tracing must see the effective input shardings, and lowering rejects staged input types that disagree
/// with the overrides. The staging entry points in [`crate::compilation`] already apply this normalization.
///
/// # Errors
///
/// Returns an error if the number of overrides differs from the number of flat inputs, an override cannot be applied
/// to an input type, or overrides are requested for external reference inputs. The reference ABI determines the
/// effective sharding of external state, so it does not accept input sharding overrides.
///
/// # Examples
///
/// ```
/// use ryft_core::{
///     ArrayIrType, ArrayType, DataType, Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding,
///     ShardingDimension,
/// };
/// use ryft_xla::Options;
/// use ryft_xla::domains::with_input_shardings;
///
/// let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto)?])?;
/// let mesh = DeviceMesh::new(logical_mesh, vec![Device::new(0, 0), Device::new(1, 0)])?;
/// let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])])?;
/// let options = Options::new(mesh).with_input_shardings(vec![sharding.clone()]);
/// let input_type = ArrayType::new_static(DataType::F32, [4]);
/// let input_types = (ArrayIrType::Array(input_type.clone()),);
/// let input_types = with_input_shardings(input_types, &options)?;
/// assert_eq!(input_types, (ArrayIrType::Array(input_type.with_sharding(sharding)?),));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn with_input_shardings<In: Parameterized<ArrayIrType>>(input_types: In, options: &Options) -> Result<In, Error> {
    if options.input_shardings.is_none() {
        return Ok(input_types);
    }
    let input_structure = input_types.parameter_structure();
    let input_types = apply_input_shardings(input_types.into_parameters().collect(), options)?;
    Ok(In::from_parameters(input_structure, input_types).map_err(ProgramError::from)?)
}

/// Validates that the staged `output_types` carry the output sharding overrides of `options` (i.e.,
/// [`Options::output_shardings`]). The staging entry points of this crate stage the overrides after tracing (i.e., a
/// `reshard` over explicit mesh axes and a `constrain_sharding` over auto mesh axes), so that calls of a staged
/// function return the output types that its executions return. Types never record placement over auto mesh axes
/// (refer to [`ArrayType::with_sharding`]), so only the placement over the other mesh axes is compared, and lowering
/// then places the outputs over auto mesh axes as the overrides request. Outputs that are not arrays are left for the
/// later array-only conversions to reject.
fn validate_output_shardings(output_types: &[ArrayIrType], options: &Options) -> Result<(), Error> {
    let Some(output_shardings) = options.output_shardings.as_deref() else {
        return Ok(());
    };
    if output_shardings.len() != output_types.len() {
        return Err(Error::Compilation(CompilationError::ShardingCountMismatch {
            kind: "output",
            expected: output_types.len(),
            actual: output_shardings.len(),
        }));
    }
    for (index, (r#type, sharding)) in output_types.iter().zip(output_shardings).enumerate() {
        let ArrayIrType::Array(array_type) = r#type else {
            continue;
        };
        if array_type.sharding() != Some(&sharding.without_auto_axes()) {
            return Err(Error::Compilation(CompilationError::InvalidOptions {
                reason: format!(
                    "staged output {index} has type `{array_type}`, which does not carry the sharding `{sharding}` \
                     requested by `output_shardings`; reshard the outputs to their overrides before staging them",
                ),
            }));
        }
    }
    Ok(())
}

/// Rejects every reference-typed output of a capture-lifted program before lowering it into an executable. The stateful
/// XLA ABI opens and commits external input references and has no protocol for turning a backend-created array into a
/// host reference with persistent identity, nor for handing an entering reference back through an output. The
/// retained reference analysis of the capture-lifted program distinguishes the escaping local allocation from the
/// forwarded external reference. References returned by nested calls inside a program never cross that ABI, so only
/// the outputs of the lowered program itself are checked.
fn validate_reference_outputs(
    program: &CoreFlatProgram<Domain<'_>>,
    capture_count: usize,
    output_types: &[ArrayIrType],
) -> Result<(), Error> {
    let Some(output_index) = output_types.iter().position(|r#type| r#type.is_reference()) else {
        return Ok(());
    };
    let analysis = program.reference_analysis(capture_count).map_err(ProgramError::from)?;
    let root = analysis.output_roots().get(output_index).copied().flatten();
    let reason = match root.map(|root| analysis.external_source(root)) {
        Some(Some(source)) => format!(
            "output {output_index} forwards the external reference bound at {source}, and the stateful XLA ABI does \
             not return forwarded references; read the reference after the call instead",
        ),
        Some(None) => format!(
            "output {output_index} is a reference allocated inside the program and cannot be returned through the \
             stateful XLA ABI, which has no protocol to turn a backend array into a host reference",
        ),
        None => format!(
            "output {output_index} is a reference whose root the reference analysis attributes to neither an input \
             nor an allocation, and the stateful XLA ABI cannot return it",
        ),
    };
    Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
}

/// Validates backend output arity and refinement of the declared logical array types.
pub(super) fn validate_output_types(declared: &[ArrayIrType], actual: &[ArrayType]) -> Result<(), Error> {
    if declared.len() != actual.len() {
        return Err(ProgramError::InvalidOutputCount { expected: declared.len(), actual: actual.len() }.into());
    }
    for (declared, actual) in declared.iter().zip(actual) {
        let declared = <&ArrayType>::try_from(declared).map_err(ProgramError::from)?;
        if !declared.is_refined_by(actual) {
            return Err(ProgramError::InvalidArgument {
                message: format!("backend output type {actual} does not refine declared type {declared}"),
            }
            .into());
        }
    }
    Ok(())
}

/// Applies an optional per-leaf sharding override to a public array IR signature, giving each overridden type the
/// override without its auto mesh axes, which types never record (refer to [`ArrayType::with_sharding`]). The runtime
/// placement that an override requests over auto mesh axes is applied by lowering instead. When overrides are provided,
/// every overridden leaf must be an array: first-class dimensions are internal SSA values, and references must be
/// discharged into explicit array state before either kind can cross the PJRT ABI. Note that without overrides the
/// signature passes through unchecked here; non-array boundary leaves are rejected by the later array-only conversions
/// on the compile path.
fn apply_signature_shardings(
    mut types: Vec<ArrayIrType>,
    shardings: Option<&[Sharding]>,
    kind: &'static str,
) -> Result<Vec<ArrayIrType>, Error> {
    let Some(shardings) = shardings else {
        return Ok(types);
    };
    if shardings.len() != types.len() {
        return Err(Error::Compilation(CompilationError::ShardingCountMismatch {
            kind,
            expected: types.len(),
            actual: shardings.len(),
        }));
    }
    for (r#type, sharding) in types.iter_mut().zip(shardings) {
        *r#type = match r#type {
            ArrayIrType::Array(array_type) => array_type.clone().with_sharding(sharding.without_auto_axes())?.into(),
            ArrayIrType::Dimension(_) | ArrayIrType::Reference(_) => {
                return Err(ProgramError::from(TypeError::invalid(format!(
                    "cannot apply an array sharding to type `{type}`",
                )))
                .into());
            }
        };
    }
    Ok(types)
}

/// Returns whether every array type that occurs anywhere in `program` has a static shape. The walk covers the whole
/// region arena, and therefore every region boundary, constant, and instruction result, recursively through attached
/// regions. First-class dimension-typed atoms are ignored because they lower to ordinary scalars; only dynamically
/// shaped *array* types make the lowered module unacceptable to the Shardy partitioner.
fn has_only_static_array_types(program: &FlatProgram) -> bool {
    program.regions().iter().all(|region| {
        region.atoms().iter().all(|atom| match atom.r#type().as_ref() {
            ArrayIrType::Array(array_type) => array_type.static_shape().is_some(),
            ArrayIrType::Dimension(_) => true,
            // The predicate answers its own question rather than relying on the lowering-time reference rejection
            // firing first: a dynamically shaped referent counts as dynamic so that partitioner selection and its
            // guards stay correct even for programs whose only dynamic type is a reference referent.
            ArrayIrType::Reference(reference_type) => reference_type.referent().static_shape().is_some(),
        })
    })
}

/// Returns whether `program` stages a `shard_map` anywhere, including inside nested regions. `shard_map` boundaries
/// are required to be static (refer to the documentation of [`ShardMapOperation`](ryft_core::ShardMapOperation)), so
/// a `shard_map` can still appear in a program that is bounded-dynamic elsewhere, which this predicate detects.
fn contains_shard_map(program: &FlatProgram) -> bool {
    program.regions().iter().any(|region| {
        region
            .instructions()
            .iter()
            .any(|instruction| matches!(instruction.operation(), Operation::ShardMap(_)))
    })
}

/// Padding contract used when admitting one staged operation to data-dependent bounded-dynamic lowering.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DataDependentPaddingDiscipline {
    /// StableHLO propagates the operation's logical dynamic dimensions without inspecting physical padding lanes.
    Propagated,

    /// Ryft delegates identity- or sentinel-masking of padding lanes to XLA's bounded-dynamic legalizer.
    XlaMasked,

    /// Ryft delegates zero-padding of contraction lanes to XLA's bounded-dynamic legalizer.
    XlaZeroPadded,

    /// Ryft explicitly materializes bounded physical operands with operation-specific padding values or masks and
    /// restores dynamic result metadata after the physical computation.
    RyftMasked,

    /// Ryft lacks sufficient semantic or execution evidence to admit this operation safely.
    Unsupported { reason: &'static str },
}

/// Returns the XLA masking contract for one reduction kind. The exhaustive match makes adding a reduction kind a
/// compile-time update to the data-dependent inventory instead of silently inheriting another kind's padding rule.
fn reduction_data_dependent_padding_discipline(kind: ReductionKind) -> DataDependentPaddingDiscipline {
    use DataDependentPaddingDiscipline::{Unsupported, XlaMasked};

    match kind {
        ReductionKind::Sum
        | ReductionKind::Product
        | ReductionKind::Max
        | ReductionKind::Min
        | ReductionKind::Any
        | ReductionKind::All => XlaMasked,
        // A log-sum-exp `reduce` expands into maximum and summation reductions over the same axes, so it inherits
        // the masking contract of exactly those two reduction kinds: XLA masks each reduction's
        // operand with that reduction's own identity, which wipes the padding lanes of the shifted exponentials
        // before they are summed. The rank-2 arm of `data_derived_padding_fixture` executes that expansion with the
        // data-derived extent on the kept axis, which is the admission's execution evidence.
        ReductionKind::LogSumExp => XlaMasked,
        ReductionKind::Mean => {
            Unsupported { reason: "mean over a dynamically sized reduction axis is not supported by the XLA lowering" }
        }
    }
}

/// Returns the XLA masking contract for one sort direction. Keeping the match exhaustive makes sort-direction
/// additions compile-forced inventory changes because ascending and descending sorts require opposite sentinels.
fn sort_data_dependent_padding_discipline(direction: SortDirection) -> DataDependentPaddingDiscipline {
    use DataDependentPaddingDiscipline::XlaMasked;

    match direction {
        SortDirection::Ascending | SortDirection::Descending => XlaMasked,
    }
}

/// Rejects every scatter combiner until its padding behavior has kind-specific execution evidence. The exhaustive
/// match deliberately forces new combiners through the same review rather than admitting them as ordinary shape-
/// preserving operations.
fn scatter_data_dependent_padding_discipline(kind: ScatterReductionKind) -> DataDependentPaddingDiscipline {
    use DataDependentPaddingDiscipline::Unsupported;

    match kind {
        ScatterReductionKind::Overwrite => Unsupported {
            reason: "overwrite-scatter padding semantics have not been verified for data-derived dimensions",
        },
        ScatterReductionKind::Add => {
            Unsupported { reason: "add-scatter padding semantics have not been verified for data-derived dimensions" }
        }
        ScatterReductionKind::Mul => Unsupported {
            reason: "multiply-scatter padding semantics have not been verified for data-derived dimensions",
        },
        ScatterReductionKind::Min => Unsupported {
            reason: "minimum-scatter padding semantics have not been verified for data-derived dimensions",
        },
        ScatterReductionKind::Max => Unsupported {
            reason: "maximum-scatter padding semantics have not been verified for data-derived dimensions",
        },
    }
}

/// Classifies every homogeneous array operation by the padding discipline required for data-derived dimensions.
/// Keeping this match exhaustive makes a newly added operation a compile-time inventory update rather than silently
/// assuming that arbitrary padding is safe.
fn array_data_dependent_padding_discipline(
    operation: &ArrayOperation<crate::experimental::ops::ArrayConstant>,
) -> DataDependentPaddingDiscipline {
    use DataDependentPaddingDiscipline::{Propagated, RyftMasked, Unsupported, XlaMasked, XlaZeroPadded};

    match operation {
        ArrayOperation::Reduce(operation) => reduction_data_dependent_padding_discipline(operation.kind()),
        ArrayOperation::Sort(operation) => sort_data_dependent_padding_discipline(operation.direction()),
        // XLA pads every input of the variadic reduction that these lower to with that input's own initial value along
        // a dynamic reduced axis (i.e., the extremum identity paired with index zero), which never beats a live element
        // and ties only when every live element equals the identity, where index zero is correct. Dynamic axes that
        // are not reduced pass through. `data_derived_padding_fixture` executes both configurations.
        ArrayOperation::ArgMax(_) | ArrayOperation::ArgMin(_) => XlaMasked,
        // Runtime-sized slice pullbacks use unique, in-bounds point updates. XLA masks inactive update lanes;
        // the compiled slice pullback fixture checks non-maximum and empty logical extents on CPU and CUDA.
        // Other window shapes, modes, and combiners retain their independent admission requirements.
        ArrayOperation::Scatter(operation)
            if operation.kind() == ScatterReductionKind::Add
                && operation.mode() == ScatterMode::PromiseInBounds
                && operation.unique_indices()
                && operation.dimensions().update_window_dimensions().is_empty() =>
        {
            XlaMasked
        }
        ArrayOperation::Scatter(operation) => scatter_data_dependent_padding_discipline(operation.kind()),
        // A prefix scan mixes positions only along its scanned axis, and type inference already requires that axis to
        // be static, so a data-derived extent can only sit on an axis the scan treats elementwise. That reasoning is
        // the same for every combining operator, and the prefix sum in `data_derived_padding_fixture` executes this arm
        // with the data-derived extent on the unscanned axis.
        ArrayOperation::Cumulative(_) => Propagated,
        ArrayOperation::Dot(_) => XlaZeroPadded,
        ArrayOperation::RaggedDot(_) => {
            Unsupported { reason: "group-size metadata may describe positions in bounded physical padding" }
        }
        ArrayOperation::ScaledDot(_) => RyftMasked,
        ArrayOperation::RngBitGenerator(_) => Unsupported {
            reason: "random generation advances state according to physical rather than logical element count",
        },
        ArrayOperation::Dropout(_) => {
            Unsupported { reason: "dropout masks are drawn according to physical rather than logical extents" }
        }
        ArrayOperation::ParallelRaggedAllToAll(_) => {
            Unsupported { reason: "ragged collective metadata may address physical padding lanes" }
        }
        ArrayOperation::CustomCall(_) => Unsupported { reason: "custom-call physical-padding semantics are opaque" },
        ArrayOperation::Print(_) => Unsupported { reason: "print exposes the physical representation of its operand" },
        // Entry guards and `cast`s narrow or widen bounded-dynamic values explicitly: the physical buffer is sliced or
        // padded, and every axis that stays dynamic gets its logical size back.
        ArrayOperation::Guard(_) | ArrayOperation::Cast(_) => RyftMasked,
        ArrayOperation::Assert(_)
        | ArrayOperation::Zero(_)
        | ArrayOperation::ZeroLike(_)
        | ArrayOperation::One(_)
        | ArrayOperation::OneLike(_)
        | ArrayOperation::Constant(_)
        | ArrayOperation::Iota(_)
        | ArrayOperation::Abs(_)
        | ArrayOperation::Neg(_)
        | ArrayOperation::Add(_)
        | ArrayOperation::Sub(_)
        | ArrayOperation::SaturatingSub(_)
        | ArrayOperation::Mul(_)
        | ArrayOperation::Div(_)
        | ArrayOperation::Sin(_)
        | ArrayOperation::Cos(_)
        | ArrayOperation::Tan(_)
        | ArrayOperation::Atan2(_)
        | ArrayOperation::Exp(_)
        | ArrayOperation::ExpM1(_)
        | ArrayOperation::Log(_)
        | ArrayOperation::Ln1p(_)
        | ArrayOperation::LogAddExp(_)
        | ArrayOperation::Sqrt(_)
        | ArrayOperation::Rsqrt(_)
        | ArrayOperation::Tanh(_)
        | ArrayOperation::Logistic(_)
        | ArrayOperation::Erf(_)
        | ArrayOperation::Pow(_)
        | ArrayOperation::Sign(_)
        | ArrayOperation::Floor(_)
        | ArrayOperation::Ceil(_)
        | ArrayOperation::Round(_)
        | ArrayOperation::Max(_)
        | ArrayOperation::Min(_)
        | ArrayOperation::Clamp(_)
        | ArrayOperation::Rem(_)
        | ArrayOperation::Not(_)
        | ArrayOperation::And(_)
        | ArrayOperation::Or(_)
        | ArrayOperation::Xor(_)
        | ArrayOperation::Complex(_)
        | ArrayOperation::Conjugate(_)
        | ArrayOperation::Real(_)
        | ArrayOperation::Imaginary(_)
        | ArrayOperation::ParallelReduce(_)
        | ArrayOperation::ParallelVary(_)
        | ArrayOperation::ParallelCast(_)
        | ArrayOperation::ParallelAllGather(_)
        | ArrayOperation::ParallelSumScatter(_)
        | ArrayOperation::ParallelPermute(_)
        | ArrayOperation::ParallelBroadcast(_)
        | ArrayOperation::ParallelAllToAll(_)
        | ArrayOperation::AxisIndex(_)
        | ArrayOperation::Reverse(_)
        | ArrayOperation::Transpose(_)
        | ArrayOperation::Reshape(_)
        | ArrayOperation::Broadcast(_)
        | ArrayOperation::Pad(_)
        | ArrayOperation::Concatenate(_)
        | ArrayOperation::Gather(_)
        | ArrayOperation::Slice(_)
        | ArrayOperation::UpdateSlice(_)
        | ArrayOperation::DynamicSlice(_)
        | ArrayOperation::DynamicUpdateSlice(_)
        | ArrayOperation::Compare(_)
        | ArrayOperation::Select(_)
        | ArrayOperation::Condition(_)
        | ArrayOperation::While(_)
        | ArrayOperation::Scan(_)
        | ArrayOperation::ConvertElementType(_)
        | ArrayOperation::ReducePrecision(_)
        | ArrayOperation::TransferToMemory(_)
        | ArrayOperation::Reshard(_)
        | ArrayOperation::ConstrainSharding(_)
        | ArrayOperation::StopGradient(_)
        | ArrayOperation::Tag(_)
        | ArrayOperation::Rematerialize(_)
        | ArrayOperation::Composite(_)
        | ArrayOperation::CompositeTranspose(_)
        | ArrayOperation::LinearCall(_) => Propagated,
    }
}

/// Classifies every XLA operation by the bounded-dynamic padding discipline it requires.
fn data_dependent_padding_discipline(operation: &Operation) -> DataDependentPaddingDiscipline {
    use DataDependentPaddingDiscipline::{Propagated, RyftMasked, Unsupported};

    match operation {
        Operation::Array(operation) => array_data_dependent_padding_discipline(operation),
        Operation::RngBitGenerator(_) => Unsupported {
            reason: "random generation advances state according to physical rather than logical element count",
        },
        Operation::ParallelRaggedAllToAll(_) => {
            Unsupported { reason: "ragged collective metadata may address physical padding lanes" }
        }
        Operation::CustomCall(_) => Unsupported { reason: "custom-call physical-padding semantics are opaque" },
        Operation::ReferenceNew(_)
        | Operation::ReferenceRead(_)
        | Operation::ReferenceWrite(_)
        | Operation::ReferenceSwap(_)
        | Operation::ReferenceAddUpdate(_)
        | Operation::ReferenceAtomicAddUpdate(_)
        | Operation::ReferenceFreeze(_) => {
            Unsupported { reason: "references must be discharged before bounded-dynamic XLA validation" }
        }
        Operation::Kernel(_) => {
            Unsupported { reason: "kernel calls must be selected before bounded-dynamic XLA validation" }
        }
        // Entry guards and `cast`s narrow or widen bounded-dynamic values explicitly (refer to the array arms).
        Operation::Guard(_) | Operation::Cast(_) => RyftMasked,
        Operation::ShardMap(_) => {
            Unsupported { reason: "shard-map lowering requires Shardy, which does not support bounded-dynamic tensors" }
        }
        Operation::Zero(_)
        | Operation::DynamicOne(_)
        | Operation::DynamicIota(_)
        | Operation::Dimension(_)
        | Operation::Compare(_)
        | Operation::Assert(_)
        | Operation::DimensionSize(_)
        | Operation::DimensionFromScalar(_)
        | Operation::DimensionToScalar(_)
        | Operation::Reshape(_)
        | Operation::Broadcast(_)
        | Operation::Concatenate(_)
        | Operation::Pad(_)
        | Operation::DynamicSlice(_)
        | Operation::ParallelAllGather(_)
        | Operation::ParallelSumScatter(_)
        | Operation::ParallelAllToAll(_)
        | Operation::Condition(_)
        | Operation::While(_)
        | Operation::Scan(_)
        | Operation::Composite(_)
        | Operation::LinearCall(_)
        | Operation::Rematerialize(_)
        | Operation::CompositeTranspose(_)
        | Operation::LiftedComposite(_)
        | Operation::LiftedCompositeTranspose(_)
        | Operation::JitCall(_) => Propagated,
    }
}

/// Returns whether `r#type` carries an identity derived from `dimension_from_scalar`.
fn references_data_derived_dimension(r#type: &ArrayIrType, variables: &HashSet<DimensionVariable>) -> bool {
    r#type.identities().any(|(_, identity)| variables.contains(identity))
}

/// Validates the bounded-dynamic compilation contract for dimensions produced by `dimension_from_scalar`.
///
/// `XlaMasked` and `XlaZeroPadded` are explicit delegation decisions: the bounded-dynamic XLA legalizer owns those
/// transformations. `RyftMasked` operations instead materialize their physical kernel inputs and restore logical
/// extents in Ryft's lowering. Only configurations with execution evidence are admitted; unbounded gateways and
/// unsupported consumers are rejected before MLIR construction.
fn validate_data_dependent_compilation(program: &FlatProgram) -> Result<(), Error> {
    let mut variables = HashSet::new();
    for region in program.regions() {
        for instruction in region.instructions() {
            if let Operation::DimensionFromScalar(operation) = instruction.operation() {
                // A static output defines no identity and is checked against its extent.
                let Some(variable) = operation.output_type().variable() else {
                    continue;
                };
                if variable.bounds().upper().is_none() {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "data-derived dimension `{variable}` with bounds {} needs a finite upper bound for XLA \
                             compilation",
                            variable.bounds(),
                        ),
                    }
                    .into());
                }
                variables.insert(variable.clone());
            }
        }
    }
    if variables.is_empty() {
        return Ok(());
    }

    // Dimension arithmetic produces fresh identities, so close the data-derived set transitively over dimension-typed
    // instruction outputs before classifying the array operations that consume those identities.
    loop {
        let count = variables.len();
        for region in program.regions() {
            for instruction in region.instructions() {
                if !instruction.inputs().iter().any(|input| {
                    references_data_derived_dimension(region.atoms()[input.index()].r#type().as_ref(), &variables)
                }) {
                    continue;
                }
                for output in instruction.outputs() {
                    if let ArrayIrType::Dimension(r#type) = region.atoms()[output.index()].r#type().as_ref() {
                        variables.insert(r#type.variable().unwrap().clone());
                    }
                }
            }
        }
        if variables.len() == count {
            break;
        }
    }

    for region in program.regions() {
        for instruction in region.instructions() {
            let consumes_data_derived_extent = instruction.inputs().iter().chain(instruction.outputs()).any(|atom| {
                references_data_derived_dimension(region.atoms()[atom.index()].r#type().as_ref(), &variables)
            });
            if consumes_data_derived_extent {
                if let DataDependentPaddingDiscipline::Unsupported { reason } =
                    data_dependent_padding_discipline(instruction.operation())
                {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "operation `{}` cannot consume data-derived dimensions during XLA lowering because \
                             {reason}",
                            instruction.operation().name(),
                        ),
                    }
                    .into());
                }
            }
        }
    }
    Ok(())
}

/// Overlays the selected multi-device execution mode on the base [`CompilationOptions`] template. Static programs
/// use Shardy SPMD partitioning, while bounded-dynamic programs use one fully replicated executable replica per
/// device because Shardy does not yet accept dynamic tensors.
pub(super) fn jit_compilation_options(
    base: &CompilationOptions,
    device_count: usize,
    use_shardy_partitioner: bool,
    feedback_directed_profile: Option<&FeedbackDirectedProfile>,
) -> CompilationOptions {
    use ryft_pjrt::protos::ExecutableCompilationOptions;
    let mut options = base.clone();
    let executable_options = options.executable_build_options.get_or_insert_with(ExecutableCompilationOptions::default);
    if executable_options.device_ordinal == 0 {
        executable_options.device_ordinal = -1;
    }
    executable_options.replica_count = if use_shardy_partitioner { 1 } else { device_count as i64 };
    executable_options.partition_count = if use_shardy_partitioner { device_count as i64 } else { 1 };
    executable_options.use_spmd_partitioning = use_shardy_partitioner;
    executable_options.use_shardy_partitioner = use_shardy_partitioner;
    if let Some(profile) = feedback_directed_profile {
        executable_options.fdo_profile = profile.bytes().to_vec();
        options.profile_version = profile.version();
    }
    options
}

/// Encodes compilation options deterministically even though Prost represents protobuf maps as randomly seeded
/// `HashMap`s. Map entries are removed from a clone before normal protobuf encoding and appended as length-delimited,
/// path-qualified records sorted by their encoded keys and values.
pub(super) fn canonical_compilation_options_bytes(options: &CompilationOptions) -> Vec<u8> {
    use ryft_pjrt::protos::{AutoTuneReferenceKey, AutoTuneResultKey};

    /// Appends one sorted protobuf-map record with an explicit path and length delimiters.
    fn append_record(output: &mut Vec<u8>, path: &[u8], mut entries: Vec<(Vec<u8>, Vec<u8>)>) {
        entries.sort_unstable();
        output.extend_from_slice(&(path.len() as u64).to_le_bytes());
        output.extend_from_slice(path);
        output.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (key, value) in entries {
            output.extend_from_slice(&(key.len() as u64).to_le_bytes());
            output.extend_from_slice(key.as_slice());
            output.extend_from_slice(&(value.len() as u64).to_le_bytes());
            output.extend_from_slice(value.as_slice());
        }
    }

    /// Removes tuning knobs from a protobuf algorithm key and appends their sorted canonical record.
    fn drain_algorithm_map(output: &mut Vec<u8>, path: &[u8], algorithm: &mut ryft_pjrt::protos::AutoTuneAlgorithmKey) {
        append_record(
            output,
            path,
            std::mem::take(&mut algorithm.tuning_knobs)
                .into_iter()
                .map(|(key, value)| (key.to_le_bytes().to_vec(), value.to_le_bytes().to_vec()))
                .collect(),
        );
    }

    let mut options = options.clone();
    let mut maps = Vec::new();
    append_record(
        &mut maps,
        b"compilation.environment_option_overrides",
        std::mem::take(&mut options.environment_option_overrides)
            .into_iter()
            .map(|(key, value)| (key.into_bytes(), value.encode_to_vec()))
            .collect(),
    );
    if let Some(debug_options) =
        options.executable_build_options.as_mut().and_then(|options| options.debug_options.as_mut())
    {
        append_record(
            &mut maps,
            b"executable.debug.analytical_latency_estimator",
            std::mem::take(&mut debug_options.xla_gpu_analytical_latency_estimator_options)
                .into_iter()
                .map(|(key, value)| (key.into_bytes(), value.into_bytes()))
                .collect(),
        );
        append_record(
            &mut maps,
            b"executable.debug.experimental_cost_model_gemm_tiling",
            std::mem::take(&mut debug_options.xla_gpu_experimental_cost_model_gemm_tiling_options)
                .into_iter()
                .map(|(key, value)| (key.into_bytes(), value.into_bytes()))
                .collect(),
        );
        append_record(
            &mut maps,
            b"executable.debug.backend_extra_options",
            std::mem::take(&mut debug_options.xla_backend_extra_options)
                .into_iter()
                .map(|(key, value)| (key.into_bytes(), value.into_bytes()))
                .collect(),
        );
    }
    if let Some(target) = options.target_config.as_mut() {
        if let Some(device) = target.gpu_device_information.as_mut() {
            for (name, description) in [
                (b"target.device.scalar_rates".as_slice(), device.scalar_unit_description.as_mut()),
                (b"target.device.matrix_rates".as_slice(), device.matrix_unit_description.as_mut()),
            ] {
                if let Some(description) = description {
                    append_record(
                        &mut maps,
                        name,
                        std::mem::take(&mut description.rate_information)
                            .into_iter()
                            .map(|(key, value)| (key.to_le_bytes().to_vec(), value.encode_to_vec()))
                            .collect(),
                    );
                }
            }
        }
        if let Some(autotune_results) = target.autotune_results.as_mut() {
            for (index, entry) in autotune_results.results.iter_mut().enumerate() {
                let Some(result) = entry.result.as_mut() else {
                    continue;
                };
                if let Some(AutoTuneResultKey::Algorithm(algorithm)) = result.key.as_mut() {
                    drain_algorithm_map(&mut maps, format!("target.autotune.{index}.algorithm").as_bytes(), algorithm);
                }
                if let Some(AutoTuneReferenceKey::ReferenceAlgorithm(algorithm)) =
                    result.failure.as_mut().and_then(|failure| failure.key.as_mut())
                {
                    drain_algorithm_map(
                        &mut maps,
                        format!("target.autotune.{index}.reference_algorithm").as_bytes(),
                        algorithm,
                    );
                }
            }
        }
    }

    let protobuf = options.encode_to_vec();
    let mut bytes = b"RYFT_XLA_COMPILATION_OPTIONS_V1".to_vec();
    bytes.extend_from_slice(&(protobuf.len() as u64).to_le_bytes());
    bytes.extend_from_slice(protobuf.as_slice());
    bytes.extend_from_slice(maps.as_slice());
    bytes
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    #[cfg(feature = "cuda-13")]
    use std::time::Instant;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    #[cfg(test)]
    use ryft_core::Array as CoreArray;
    #[cfg(feature = "cuda-13")]
    use ryft_core::dot_product_attention;
    use ryft_core::{
        AddOperation, ArgMaxOperation, ArgMinOperation, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation,
        ArrayReferenceTransform, ArrayReferenceTransformIndex, ArraySliceAxis, ArrayType, AtomId,
        AttentionConfiguration, AttentionImplementation, AttentionInputs, AxisIndexOperation, BatchAxis,
        CaptureReference, CollectiveOptions, CompareOperation, ComparisonDirection, CompilationCacheDomain,
        CompilationDomain, CompilationTracer, Compile, CompiledFunctionDispatcher, ConditionOperation,
        ConstantOperation, Context, ConvertElementTypeOperation, CotangentDestinationKind, CumulativeKind,
        CumulativeOperation, CustomCallOperation, DataType, Device, DeviceMesh, Differentiate,
        Dimension as CoreDimension, DimensionAddOperation, DimensionBounds, DimensionFromScalar,
        DimensionFromScalarOperation, DimensionOperation, DimensionSizeOperation, DimensionToScalarOperation,
        DimensionType, DimensionValue, DimensionVariable, DomainTracingContext, DotDimensionNumbers, DotOperation,
        DynamicBroadcastOperation, DynamicReshapeOperation, DynamicSliceOperation, DynamicSliceWithDimensions,
        IotaOperation, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, MulOperation, NegOperation, OneOperation,
        Operation as CoreOperation, OperationPayloadProjection, ParallelAllGatherOperation,
        ParallelAllGatherOutputVariance, ParallelAllToAllOperation, ParallelPermuteOperation, ParallelReduceOperation,
        ParallelVaryOperation, Placeholder, PrintOperation, Program as CoreProgram,
        ProgramBuilder as CoreProgramBuilder, ProgramError, Provenance, ProvenanceScope, RaggedDotDimensionNumbers,
        RaggedDotOperation, RandomAlgorithm, ReduceOperation, ReductionKind, ReferenceAddUpdate,
        ReferenceAddUpdateOperation, ReferenceFreeze, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation,
        ReferenceRead, ReferenceReadOperation, ReferenceSwapOperation, ReferenceType, ReferenceWrite,
        ReferenceWriteOperation, ReshapeOperation, Reshard, RngBitGeneratorOperation, ScaledDotOperation,
        ScanOperation, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind, Shape, ShardMapBoundary,
        ShardMapOperation, ShardMapOptions, Sharding, ShardingDimension, SliceOperation, SortDirection, SortOperation,
        StagedFunction as CoreStagedFunction, StagingContext, StaticShape, SubOperation, TiledLayout, TracingContext,
        Type, Typed, TypedProgram, Value as CoreValue, ValueProjection, WhileOperation, batch,
        differentiable_dot_product_attention, shard_map,
    };
    use ryft_pjrt::protos::CompilationOptions;
    use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{GpuClientOptions, GpuMemoryAllocator, GpuPlatform, load_cuda_13_plugin};

    use crate::arrays_v0::host::materialize_dense_array_bytes;
    use crate::domains::Tracer;
    use crate::domains::persistence::PERSISTENT_KEY_SCHEMA_VERSION;
    use crate::domains::sessions::Session;
    use crate::domains::tests::*;
    use crate::experimental::ops::{Constant, FlatProgram, Operation, ProgramBuilder};
    use crate::experimental::tracing::ArrayTracer;
    use crate::operations::jit_call::JitCallOperation;
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{Array, Error, FromPjrt, RaggedDotLoweringStrategy, ToPjrt, Value};

    use super::*;

    /// Reads the first addressable shard as `bfloat16` values converted to `f32`.
    fn read_bf16s(client: &Client<'_>, array: &Array<'_>) -> Vec<f32> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<u16>(bytes.as_slice())
            .into_iter()
            .map(|bits| half::bf16::from_bits(bits).to_f32())
            .collect()
    }

    /// Reads the first addressable shard as native-endian signed 32-bit integers.
    fn read_i32s(client: &Client<'_>, array: &Array<'_>) -> Vec<i32> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<i32>(bytes.as_slice())
    }

    /// Builds the kind-sensitive bounded-dynamic padding fixture shared by CPU and CUDA execution tests.
    fn data_derived_padding_fixture(size_type: ArrayType, extent: DimensionVariable) -> FlatProgram {
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let index_type = dynamic_type.clone().with_data_type(DataType::U64);
        let matrix_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![CoreDimension::Dynamic(extent.clone()), CoreDimension::Static(4)]),
        );

        // The iota and its negation make physical suffix lanes numerically distinguishable from every admitted live
        // prefix. This catches both reduction identities and the opposite sentinels needed by ascending and descending
        // sort. The boolean reductions cover their own false/true identities, and dot covers zero-padded contraction.
        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let iota = builder
            .add_instruction(IotaOperation::new(dynamic_type.clone(), 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let indices = builder
            .add_instruction(IotaOperation::new(index_type, 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let negative = builder.add_instruction(NegOperation::new(), Vec::new(), vec![iota], None).unwrap()[0];
        let sorted_ascending = builder
            .add_instruction(SortOperation::new(0, SortDirection::Ascending), Vec::new(), vec![negative], None)
            .unwrap()[0];
        let sorted_descending = builder
            .add_instruction(SortOperation::new(0, SortDirection::Descending), Vec::new(), vec![iota], None)
            .unwrap()[0];
        let argmin_indices = builder
            .add_instruction(SortOperation::new(0, SortDirection::Ascending), Vec::new(), vec![negative, indices], None)
            .unwrap()[1];
        let argmax_indices = builder
            .add_instruction(SortOperation::new(0, SortDirection::Descending), Vec::new(), vec![iota, indices], None)
            .unwrap()[1];
        let sum = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![iota], None)
            .unwrap()[0];
        // Shift the iota to nonzero factors so including physical padding changes the product.
        let ones =
            builder.add_instruction(OneOperation::new(dynamic_type), Vec::new(), vec![dimension], None).unwrap()[0];
        let factors = builder.add_instruction(AddOperation::new(), Vec::new(), vec![iota, ones], None).unwrap()[0];
        let product = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Product), Vec::new(), vec![factors], None)
            .unwrap()[0];
        let maximum = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Max), Vec::new(), vec![iota], None)
            .unwrap()[0];
        let minimum = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Min), Vec::new(), vec![negative], None)
            .unwrap()[0];
        let any_input = builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::<ArrayType>::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![negative, iota],
                None,
            )
            .unwrap()[0];
        let any = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Any), Vec::new(), vec![any_input], None)
            .unwrap()[0];
        let all_input = builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::<ArrayType>::new(ComparisonDirection::GreaterThanOrEqual)),
                Vec::new(),
                vec![iota, negative],
                None,
            )
            .unwrap()[0];
        let all = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::All), Vec::new(), vec![all_input], None)
            .unwrap()[0];
        let dot = builder
            .add_instruction(
                DotOperation::new(DotDimensionNumbers::inner_product()),
                Vec::new(),
                vec![iota, iota],
                None,
            )
            .unwrap()[0];

        // The rank-2 `i + j` matrix carries the data-derived extent on an axis that neither the reduction nor the scan
        // touches, which is the configuration those two disciplines actually admit: a log-sum-exp `reduce` consumes
        // the static axis and still produces a dynamically shaped output, and the prefix scan consumes the static
        // axis while every row it emits keeps the dynamic extent.
        let rows = builder
            .add_instruction(IotaOperation::new(matrix_type.clone(), 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let columns = builder
            .add_instruction(IotaOperation::new(matrix_type, 1).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let matrix = builder.add_instruction(AddOperation::new(), Vec::new(), vec![rows, columns], None).unwrap()[0];
        let log_sum_exp = builder
            .add_instruction(ReduceOperation::new(vec![1], ReductionKind::LogSumExp), Vec::new(), vec![matrix], None)
            .unwrap()[0];
        let cumulative_sum = builder
            .add_instruction(CumulativeOperation::new(1, CumulativeKind::Sum), Vec::new(), vec![matrix], None)
            .unwrap()[0];

        // Index reductions over the data-derived axis would select a physical suffix lane if its padding were not
        // masked, because the suffix lanes of the increasing iota are larger than every live lane and those of its
        // negation are smaller. The matrix covers both a reduced data-derived axis and a data-derived axis that the
        // reduction keeps.
        let argmax = builder
            .add_instruction(ArgMaxOperation::new(0, DataType::I32), Vec::new(), vec![iota], None)
            .unwrap()[0];
        let argmin = builder
            .add_instruction(ArgMinOperation::new(0, DataType::I32), Vec::new(), vec![negative], None)
            .unwrap()[0];
        let row_argmax = builder
            .add_instruction(ArgMaxOperation::new(1, DataType::I32), Vec::new(), vec![matrix], None)
            .unwrap()[0];
        let column_argmax = builder
            .add_instruction(ArgMaxOperation::new(0, DataType::I32), Vec::new(), vec![matrix], None)
            .unwrap()[0];
        builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![
                    sorted_ascending,
                    sorted_descending,
                    argmin_indices,
                    argmax_indices,
                    sum,
                    maximum,
                    minimum,
                    any,
                    all,
                    dot,
                    log_sum_exp,
                    cumulative_sum,
                    product,
                    argmax,
                    argmin,
                    row_argmax,
                    column_argmax,
                ],
                vec![Placeholder],
                vec![Placeholder; 17],
            )
            .unwrap()
    }

    /// Builds the bounded-dynamic scaled-dot and attention proof fixture. The attention values vary along the
    /// key/value sequence axis, so including any physical suffix lane changes every valid output row. Attention is
    /// staged by the training function with the provided implementation, whose reverse mode differentiation calls the
    /// forward attention definition with its residual and the backward attention definition.
    fn data_derived_scaled_dot_attention_fixture(
        size_type: ArrayType,
        extent: DimensionVariable,
        scaled_element_type: DataType,
        scaled_scale_type: DataType,
        attention_implementation: AttentionImplementation,
    ) -> FlatProgram {
        /// Binds `operation` to `inputs` in the context of `value` and returns its only output.
        fn bind<O: Into<Operation>>(
            value: &Tracer<'static>,
            operation: O,
            inputs: &[Tracer<'static>],
        ) -> Tracer<'static> {
            value.domain().bind(operation, Vec::<FlatProgram>::new(), inputs).unwrap().remove(0)
        }

        let scaled_block_size = match scaled_element_type {
            DataType::F8E4M3FN | DataType::F8E5M2 => 32,
            _ => 16,
        };
        let attention_batch = DimensionVariable::new("attention_batch", extent.bounds().clone());
        let scaled_lhs_type = ArrayType::new(
            scaled_element_type,
            Shape::new(vec![CoreDimension::Dynamic(extent.clone()), CoreDimension::Static(scaled_block_size)]),
        );
        let scaled_lhs_scale_type = ArrayType::new(
            scaled_scale_type,
            Shape::new(vec![CoreDimension::Dynamic(extent.clone()), CoreDimension::Static(1)]),
        );
        let scaled_rhs_type = ArrayType::new(
            scaled_element_type,
            Shape::new(vec![CoreDimension::Static(1), CoreDimension::Static(scaled_block_size)]),
        );
        let scaled_rhs_scale_type =
            ArrayType::new(scaled_scale_type, Shape::new(vec![CoreDimension::Static(1), CoreDimension::Static(1)]));
        let attention_type = ArrayType::new(
            DataType::BF16,
            Shape::new(vec![
                CoreDimension::Dynamic(attention_batch.clone()),
                CoreDimension::Dynamic(extent.clone()),
                CoreDimension::Static(1),
                CoreDimension::Static(8),
            ]),
        );
        let attention = differentiable_dot_product_attention::<Domain<'static>>(
            AttentionConfiguration::new().with_scale(1.0).with_implementation(attention_implementation),
        );
        let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
                let size = &inputs[0];
                let dimension = bind(size, DimensionFromScalarOperation::new(extent.clone().into()), &[size.clone()]);
                let batch_dimension =
                    bind(size, DimensionFromScalarOperation::new(attention_batch.clone().into()), &[size.clone()]);

                // `scaled_dot`: the runtime row extent is non-contracting. Both dynamic element/scale operands share
                // it, while the static right operand fixes one output column and the static block shape fixes
                // `k = 16`.
                let scaled_lhs = bind(size, OneOperation::new(scaled_lhs_type.clone()), &[dimension.clone()]);
                let scaled_lhs_scales =
                    bind(size, OneOperation::new(scaled_lhs_scale_type.clone()), &[dimension.clone()]);
                let scaled_rhs = bind(size, OneOperation::new(scaled_rhs_type.clone()), &[]);
                let scaled_rhs_scales = bind(size, OneOperation::new(scaled_rhs_scale_type.clone()), &[]);
                let scaled_dot = bind(
                    size,
                    ScaledDotOperation::new(
                        DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                        DataType::F32,
                        true,
                        true,
                    ),
                    &[scaled_lhs, scaled_rhs, scaled_lhs_scales, scaled_rhs_scales],
                );

                // Attention: equal query/key scores make the live softmax uniform, while values equal their sequence
                // index. Explicit lengths of `extent - 1` exercise the supported further-restriction contract: visible
                // query rows return the mean `(extent - 2) / 2`, while the final logical row and all physical suffix
                // lanes stay zero. Backward has exact `dQ = 0`, `dK[j] = 8 * (j - mean)`, and `dV = 1` on visible
                // rows and zeros elsewhere.
                let extents = [batch_dimension.clone(), dimension.clone()];
                let query = bind(size, OneOperation::new(attention_type.clone()), &extents);
                let key = bind(size, OneOperation::new(attention_type.clone()), &extents);
                let value = bind(size, IotaOperation::new(attention_type.clone(), 1).unwrap(), &extents);
                let output_cotangent = bind(size, OneOperation::new(attention_type.clone()), &extents);
                let logical_length = bind(size, DimensionToScalarOperation, &[dimension.clone()]);
                let logical_length =
                    bind(size, ConvertElementTypeOperation::<ArrayType>::new(DataType::I32, false), &[logical_length]);
                let one = CoreArray::from_elements(ArrayType::scalar(DataType::I32), &[1i32]).unwrap();
                let one = bind(size, ConstantOperation::new(one), &[]);
                let length = bind(size, SubOperation::new(), &[logical_length, one]);
                let lengths =
                    bind(size, DynamicBroadcastOperation::new(Vec::new()), &[length, batch_dimension.clone()]);
                let inputs = AttentionInputs {
                    query_sequence_lengths: Some(lengths.clone()),
                    key_value_sequence_lengths: Some(lengths),
                    ..AttentionInputs::new(query, key, value)
                };
                let (output, pullback) = size.domain().differentiate_at(inputs).vjp(|inputs| attention.call(inputs))?;
                let cotangents = pullback.apply_materialized(output_cotangent)?;
                Ok(vec![scaled_dot, output, cotangents.query, cotangents.key, cotangents.value])
            },
            vec![ArrayIrType::from(size_type)],
        )
        .unwrap();
        program.into_flat_program()
    }

    /// Kind of `shard_map` program that [`shard_map_differential_program`] builds to compare the reference backend's
    /// emulation of `shard_map` with its execution through XLA.
    #[derive(Copy, Clone, Debug)]
    enum ShardMapDifferentialCase {
        /// Tiled identity of an input sharded along the manual axis `x`, together with its sum across `x`.
        IdentityAndSum,

        /// Coordinate of every device along the manual axis `x`.
        AxisIndex,

        /// Invariant tiled all-gather across the manual axis `x`.
        AllGather,

        /// Tiled all-to-all across the manual axis `x` that moves a row-sharded matrix to a column-sharded one.
        AllToAll,

        /// Permutation across the manual axis `x` that sends the shard of device 0 to device 1 and zeros to device 0.
        Permute,

        /// Map over the manual axis `x` whose body maps over the manual axis `y` and sums across the enclosing axis
        /// `x`.
        Nested,

        /// Map that adds its shard of an update into its shard of a caller-allocated reference and reads it back.
        Reference,

        /// Map that reads a reference replicated along the manual axis `x` at the coordinate of every device along `x`.
        VaryingIndexReference,

        /// Map that doubles a reference replicated along the manual axis `x` on every device and reads it back.
        ReplicatedReference,
    }

    /// Builds a program over `input_types` whose outputs are the atoms that `build` adds to it.
    fn build_differential_program<V: CoreValue<Type = ArrayIrType>, O: CoreOperation<Type = ArrayIrType>>(
        input_types: Vec<ArrayIrType>,
        build: impl FnOnce(&mut CoreProgramBuilder<V, O>, &[AtomId]) -> Vec<AtomId>,
    ) -> CoreProgram<V, O, Vec<V>, Vec<V>> {
        let mut builder = CoreProgramBuilder::<V, O>::new();
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let outputs = build(&mut builder, &inputs);
        let output_count = outputs.len();
        builder.build(outputs, vec![Placeholder; inputs.len()], vec![Placeholder; output_count]).unwrap()
    }

    /// Builds the program of `case` over any operation family `O` that lifts the array IR family over `A`, so that the
    /// reference backend's emulation (over [`ArrayIrOperation<Array>`]) and XLA execution (over [`Operation`]) run
    /// the same program. Every program input and output is an array: the reference of
    /// [`ShardMapDifferentialCase::Reference`] is allocated from a program input, and its final value is frozen into a
    /// trailing program output.
    fn shard_map_differential_program<A, V, O>(case: ShardMapDifferentialCase) -> CoreProgram<V, O, Vec<V>, Vec<V>>
    where
        A: CoreValue<Type = ArrayType>,
        V: CoreValue<Type = ArrayIrType>,
        O: CoreOperation<Type = ArrayIrType> + OperationPayloadProjection + From<ArrayIrOperation<A>>,
    {
        let array_operation = ArrayIrOperation::<A>::from;
        let mesh = match case {
            ShardMapDifferentialCase::Nested => LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
                MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            ]),
            _ => LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]),
        }
        .unwrap();
        let sharded = |axis: &str| Sharding::new(mesh.clone(), vec![ShardingDimension::sharded([axis])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let f32_type = |shape: &[usize], sharding: &Sharding| {
            ArrayType::new_static(DataType::F32, shape.to_vec()).with_sharding(sharding.clone()).unwrap()
        };
        let sum_over_x = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone());
        let (boundary, global_input_types, body) = match case {
            ShardMapDifferentialCase::IdentityAndSum => {
                let boundary = ShardMapBoundary::new(
                    mesh.clone(),
                    vec![sharded("x")],
                    vec![sharded("x"), replicated],
                    vec!["x".into()],
                )
                .unwrap();
                let global_type = f32_type(&[4], &sharded("x"));
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, inputs| {
                    let sum = array_operation(ArrayOperation::ParallelReduce(sum_over_x));
                    vec![inputs[0], builder.add_instruction(sum, Vec::new(), vec![inputs[0]], None).unwrap()[0]]
                });
                (boundary, vec![global_type.into()], body)
            }
            ShardMapDifferentialCase::AxisIndex => {
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![sharded("x")], vec![sharded("x")], vec!["x".into()])
                        .unwrap();
                let global_type = f32_type(&[2], &sharded("x"));
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, _| {
                    let index = AxisIndexOperation::new("x".to_string()).with_mesh(mesh.clone());
                    let index = array_operation(ArrayOperation::AxisIndex(index));
                    let index = builder.add_instruction(index, Vec::new(), Vec::new(), None).unwrap()[0];
                    let reshape = ReshapeOperation::new(Shape::new(vec![CoreDimension::Static(1)]));
                    let reshape = array_operation(ArrayOperation::Reshape(reshape));
                    builder.add_instruction(reshape, Vec::new(), vec![index], None).unwrap().to_vec()
                });
                (boundary, vec![global_type.into()], body)
            }
            ShardMapDifferentialCase::AllGather => {
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![sharded("x")], vec![replicated], vec!["x".into()])
                        .unwrap();
                let global_type = f32_type(&[4], &sharded("x"));
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, inputs| {
                    let gather = ParallelAllGatherOperation::new(
                        "x".to_string(),
                        2,
                        0,
                        CollectiveOptions::tiled(),
                        ParallelAllGatherOutputVariance::Invariant,
                    );
                    let gather = array_operation(ArrayOperation::ParallelAllGather(gather.with_mesh(mesh.clone())));
                    builder.add_instruction(gather, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
                });
                (boundary, vec![global_type.into()], body)
            }
            ShardMapDifferentialCase::AllToAll => {
                let rows =
                    Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]);
                let columns =
                    Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])]);
                let (rows, columns) = (rows.unwrap(), columns.unwrap());
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![rows.clone()], vec![columns], vec!["x".into()]).unwrap();
                let global_type = f32_type(&[2, 4], &rows);
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, inputs| {
                    let exchange = ParallelAllToAllOperation::new("x".to_string(), 2, 1, 0, CollectiveOptions::tiled());
                    let exchange = array_operation(ArrayOperation::ParallelAllToAll(exchange.with_mesh(mesh.clone())));
                    builder.add_instruction(exchange, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
                });
                (boundary, vec![global_type.into()], body)
            }
            ShardMapDifferentialCase::Permute => {
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![sharded("x")], vec![sharded("x")], vec!["x".into()])
                        .unwrap();
                let global_type = f32_type(&[4], &sharded("x"));
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, inputs| {
                    let permute = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]);
                    let permute = array_operation(ArrayOperation::ParallelPermute(permute.with_mesh(mesh.clone())));
                    builder.add_instruction(permute, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
                });
                (boundary, vec![global_type.into()], body)
            }
            ShardMapDifferentialCase::Nested => {
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![sharded("x")], vec![replicated], vec!["x".into()])
                        .unwrap();
                let global_type = f32_type(&[4], &sharded("x"));
                let outer_local_type = boundary.local_input_type(0, &global_type).unwrap();
                let inner_boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![sharded("y")], vec![sharded("y")], vec!["y".into()])
                        .unwrap();
                let inner_local_type = inner_boundary.local_input_type(0, &outer_local_type).unwrap();
                let inner_body =
                    build_differential_program::<V, O>(vec![inner_local_type.into()], |builder, inputs| {
                        let sum = array_operation(ArrayOperation::ParallelReduce(sum_over_x));
                        builder.add_instruction(sum, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
                    });
                let inner_operation =
                    ShardMapOperation::from_program(&inner_body, vec![outer_local_type.clone().into()], inner_boundary)
                        .unwrap();
                let body = build_differential_program::<V, O>(vec![outer_local_type.into()], |builder, inputs| {
                    let inner_body = builder.import_program(inner_body);
                    let inner_operation = ArrayIrOperation::<A>::from(inner_operation);
                    builder.add_instruction(inner_operation, vec![inner_body], inputs.to_vec(), None).unwrap().to_vec()
                });
                (boundary, vec![global_type.into()], body)
            }
            ShardMapDifferentialCase::Reference => {
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![sharded("x"); 2], vec![sharded("x")], vec!["x".into()])
                        .unwrap();
                let global_type = f32_type(&[4], &sharded("x"));
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let input_types = vec![ReferenceType::new(local_type.clone()).into(), local_type.into()];
                let body = build_differential_program::<V, O>(input_types, |builder, inputs| {
                    let add_update =
                        ReferenceAddUpdateOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                    let add_update = ArrayIrOperation::<A>::from(add_update);
                    builder.add_instruction(add_update, Vec::new(), inputs.to_vec(), None).unwrap();
                    let read = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                    let read = ArrayIrOperation::<A>::from(read);
                    builder.add_instruction(read, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
                });
                (boundary, vec![ReferenceType::new(global_type.clone()).into(), global_type.into()], body)
            }
            ShardMapDifferentialCase::VaryingIndexReference => {
                let table_sharding = Sharding::replicated(mesh.clone(), 2);
                let boundary = ShardMapBoundary::new(
                    mesh.clone(),
                    vec![table_sharding.clone()],
                    vec![sharded("x")],
                    vec!["x".into()],
                )
                .unwrap();
                let global_type = f32_type(&[2, 3], &table_sharding);
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let input_types = vec![ReferenceType::new(local_type).into()];
                let body = build_differential_program::<V, O>(input_types, |builder, inputs| {
                    let index = AxisIndexOperation::new("x".to_string()).with_mesh(mesh.clone());
                    let index = array_operation(ArrayOperation::AxisIndex(index));
                    let index = builder.add_instruction(index, Vec::new(), Vec::new(), None).unwrap()[0];
                    let read = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new()
                        .with_transforms(vec![ArrayReferenceTransform::Index {
                            axis: 0,
                            index: ArrayReferenceTransformIndex::Dynamic,
                        }]);
                    let read = ArrayIrOperation::<A>::from(read);
                    builder.add_instruction(read, Vec::new(), vec![inputs[0], index], None).unwrap().to_vec()
                });
                (boundary, vec![ReferenceType::new(global_type).into()], body)
            }
            ShardMapDifferentialCase::ReplicatedReference => {
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![replicated.clone()], vec![sharded("x")], vec!["x".into()])
                        .unwrap();
                let global_type = f32_type(&[2], &replicated);
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let input_types = vec![ReferenceType::new(local_type).into()];
                let body = build_differential_program::<V, O>(input_types, |builder, inputs| {
                    let read = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                    let read = ArrayIrOperation::<A>::from(read);
                    let state = builder.add_instruction(read.clone(), Vec::new(), vec![inputs[0]], None).unwrap()[0];
                    let add_update =
                        ReferenceAddUpdateOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                    let add_update = ArrayIrOperation::<A>::from(add_update);
                    builder.add_instruction(add_update, Vec::new(), vec![inputs[0], state], None).unwrap();
                    let state = builder.add_instruction(read, Vec::new(), vec![inputs[0]], None).unwrap()[0];
                    let vary = array_operation(ArrayOperation::ParallelVary(ParallelVaryOperation::new("x".into())));
                    builder.add_instruction(vary, Vec::new(), vec![state], None).unwrap().to_vec()
                });
                (boundary, vec![ReferenceType::new(global_type).into()], body)
            }
        };
        let operation = ShardMapOperation::from_program(&body, global_input_types, boundary).unwrap();

        // Reference inputs of the map are allocated from the corresponding program inputs and frozen after the map.
        let program_input_types = operation
            .global_input_types()
            .iter()
            .map(|r#type| match r#type {
                ArrayIrType::Reference(reference) => ArrayIrType::Array(reference.referent().clone()),
                r#type => r#type.clone(),
            })
            .collect();
        build_differential_program::<V, O>(program_input_types, |builder, inputs| {
            let mut map_inputs = inputs.to_vec();
            let mut references = Vec::new();
            for (input, r#type) in map_inputs.iter_mut().zip(operation.global_input_types()) {
                if r#type.is_reference() {
                    let new = ArrayIrOperation::<A>::from(ReferenceNewOperation::<ArrayType, ArrayIrType>::new());
                    *input = builder.add_instruction(new, Vec::new(), vec![*input], None).unwrap()[0];
                    references.push(*input);
                }
            }
            let body = builder.import_program(body);
            let operation = ArrayIrOperation::<A>::from(operation);
            let mut outputs = builder.add_instruction(operation, vec![body], map_inputs, None).unwrap().to_vec();
            for reference in references {
                let freeze = ArrayIrOperation::<A>::from(ReferenceFreezeOperation::<ArrayType, ArrayIrType>::new());
                outputs.push(builder.add_instruction(freeze, Vec::new(), vec![reference], None).unwrap()[0]);
            }
            outputs
        })
    }

    /// Runs the program of `case` (refer to [`shard_map_differential_program`]) through the reference backend's
    /// emulation and through XLA execution on all devices of `client`, which must match the devices of the mesh of
    /// `case`, over the same integer-valued inputs, and checks that both produce exactly the same outputs, including
    /// their types.
    fn assert_shard_map_emulation_matches_execution(client: &Client<'_>, case: ShardMapDifferentialCase) {
        type ArrayMember = <Constant as ValueProjection<ArrayType>>::Projected;
        let emulated_program =
            shard_map_differential_program::<CoreArray, ArrayIrValue<CoreArray>, ArrayIrOperation<CoreArray>>(case);
        let xla_program = shard_map_differential_program::<ArrayMember, Constant, Operation>(case);
        let input_types = xla_program
            .input_types()
            .into_owned()
            .into_iter()
            .map(|r#type| match r#type {
                ArrayIrType::Array(r#type) => r#type,
                r#type => panic!("differential programs have array inputs only but got `{type}`"),
            })
            .collect::<Vec<_>>();
        let input_bytes = input_types
            .iter()
            .enumerate()
            .map(|(index, r#type)| {
                let element_count = r#type.static_shape().unwrap().dimensions().iter().product::<usize>();
                let elements = (0..element_count).map(|element| (100 * index + element + 1) as f32).collect::<Vec<_>>();
                values_to_bytes::<f32>(&elements)
            })
            .collect::<Vec<_>>();

        // The reference backend emulates every device of the map on the host.
        let emulated_inputs = input_types
            .iter()
            .zip(&input_bytes)
            .map(|(r#type, bytes)| ArrayIrValue::Array(CoreArray::from_logical_bytes(r#type.clone(), bytes).unwrap()))
            .collect();
        let emulated_outputs = emulated_program.interpret(emulated_inputs).unwrap();

        // XLA compiles the map into one manual computation and executes it on the devices of `client`.
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
        let mesh = DeviceMesh::new(input_types[0].sharding().unwrap().mesh().clone(), devices).unwrap();
        let domain = Session::new(client).domain();
        let inputs = input_types
            .into_iter()
            .zip(input_bytes)
            .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap())
            .collect();
        let lowered = domain.lower_program(&xla_program, 0, &Options::new(mesh)).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let outputs = domain.execute_program(&compiled, inputs).unwrap();
        let outputs = outputs
            .iter()
            .map(|output| {
                let bytes = materialize_dense_array_bytes(output).unwrap();
                ArrayIrValue::Array(CoreArray::from_logical_bytes(output.r#type().into_owned(), &bytes).unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs, emulated_outputs, "`{case:?}` outputs differ between XLA and the emulation");
    }

    /// Linearizes the program of `case` (refer to [`shard_map_differential_program`]) with respect to its first input
    /// and runs its primal program and then its pullback through the reference backend's emulation and through XLA
    /// execution on all devices of `client`, which must match the devices of the mesh of `case`, over the same
    /// integer-valued inputs and output seeds, and checks that both produce exactly the same input cotangent, including
    /// its type. The pullback consumes the residuals of the primal program of the same backend.
    fn assert_shard_map_pullback_emulation_matches_execution(client: &Client<'_>, case: ShardMapDifferentialCase) {
        type ArrayMember = <Constant as ValueProjection<ArrayType>>::Projected;
        let emulated_program =
            shard_map_differential_program::<CoreArray, ArrayIrValue<CoreArray>, ArrayIrOperation<CoreArray>>(case);
        let xla_program = shard_map_differential_program::<ArrayMember, Constant, Operation>(case);
        let emulated = emulated_program.entry_region_ref().linearize(&[0]).unwrap();
        let xla = xla_program.entry_region_ref().linearize(&[0]).unwrap();
        let array_types = |types: Vec<ArrayIrType>| {
            types
                .into_iter()
                .map(|r#type| match r#type {
                    ArrayIrType::Array(r#type) => r#type,
                    r#type => panic!("differential programs have array inputs and outputs only but got `{type}`"),
                })
                .collect::<Vec<_>>()
        };
        let input_types = array_types(xla_program.input_types().into_owned());
        let output_types = array_types(xla_program.output_types().into_owned());
        let bytes = |offset: usize, r#type: &ArrayType| {
            let element_count = r#type.static_shape().unwrap().dimensions().iter().product::<usize>();
            let elements = (0..element_count).map(|element| (offset + element + 1) as f32).collect::<Vec<_>>();
            values_to_bytes::<f32>(&elements)
        };
        let input_bytes =
            input_types.iter().enumerate().map(|(index, r#type)| bytes(100 * index, r#type)).collect::<Vec<_>>();
        let seed_bytes = output_types
            .iter()
            .enumerate()
            .map(|(index, r#type)| bytes(1000 + 100 * index, r#type))
            .collect::<Vec<_>>();

        // The reference backend emulates every device of both maps on the host.
        let emulated_values = |types: &[ArrayType], bytes: &[Vec<u8>]| {
            types
                .iter()
                .zip(bytes)
                .map(|(r#type, bytes)| {
                    ArrayIrValue::Array(CoreArray::from_logical_bytes(r#type.clone(), bytes).unwrap())
                })
                .collect::<Vec<_>>()
        };
        let mut primal_outputs = emulated.primal().interpret(emulated_values(&input_types, &input_bytes)).unwrap();
        let residuals = primal_outputs.split_off(primal_outputs.len() - emulated.residual_count());
        let mut pullback_inputs = emulated_values(&output_types, &seed_bytes);
        pullback_inputs.extend(residuals);
        let emulated_cotangents = emulated.pullback().unwrap().program().interpret(pullback_inputs).unwrap();

        // XLA compiles each map into one manual computation and executes it on the devices of `client`.
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
        let mesh = DeviceMesh::new(input_types[0].sharding().unwrap().mesh().clone(), devices).unwrap();
        let domain = Session::new(client).domain();
        let xla_values = |types: &[ArrayType], bytes: &[Vec<u8>]| {
            types
                .iter()
                .zip(bytes)
                .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type.clone(), mesh.clone(), bytes).unwrap())
                .collect::<Vec<_>>()
        };
        let execute = |program: &FlatProgram, inputs: Vec<_>| {
            let lowered = domain.lower_program(program, 0, &Options::new(mesh.clone())).unwrap();
            let compiled = domain.compile_program(&lowered).unwrap();
            domain.execute_program(&compiled, inputs).unwrap()
        };
        let mut primal_outputs = execute(xla.primal(), xla_values(&input_types, &input_bytes));
        let residuals = primal_outputs.split_off(primal_outputs.len() - xla.residual_count());
        let mut pullback_inputs = xla_values(&output_types, &seed_bytes);
        pullback_inputs.extend(residuals);
        let cotangents = execute(&xla.pullback().unwrap().into_program(), pullback_inputs)
            .iter()
            .map(|cotangent| {
                let bytes = materialize_dense_array_bytes(cotangent).unwrap();
                ArrayIrValue::Array(CoreArray::from_logical_bytes(cotangent.r#type().into_owned(), &bytes).unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(cotangents, emulated_cotangents, "`{case:?}` cotangents differ between XLA and the emulation");
    }

    #[test]
    fn test_production_array_ir_lowering_executes_dynamic_dimension_arithmetic_broadcast_and_reshape() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let build_program = |vector_type: &ArrayType, increment: usize| {
            let mut builder = ProgramBuilder::new();
            let vector = builder.add_input(vector_type.clone().into());
            let scalar = builder.add_input(scalar_type.clone().into());
            let size_operation = DimensionSizeOperation::new(vector_type, 0).unwrap();
            let size = builder.add_instruction(size_operation.clone(), Vec::new(), vec![vector], None).unwrap()[0];
            let increment_value = DimensionValue::constant(increment).unwrap();
            let increment = builder
                .add_instruction(
                    Operation::Dimension(DimensionOperation::Constant(ConstantOperation::new(increment_value.clone()))),
                    Vec::new(),
                    Vec::new(),
                    None,
                )
                .unwrap()[0];
            let add =
                DimensionAddOperation::new(size_operation.output_type(), increment_value.r#type().as_ref()).unwrap();
            let output_extent = builder
                .add_instruction(
                    Operation::Dimension(DimensionOperation::Add(add)),
                    Vec::new(),
                    vec![size, increment],
                    None,
                )
                .unwrap()[0];
            let broadcast = builder
                .add_instruction(
                    DynamicBroadcastOperation::new(Vec::new()),
                    Vec::new(),
                    vec![scalar, output_extent],
                    None,
                )
                .unwrap()[0];
            let reshaped = builder
                .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![broadcast, output_extent], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![reshaped],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };

        let dynamic_program = build_program(&vector_type, 1);
        let dynamic_lowering = domain.lower_program(&dynamic_program, 0, &Options::new(mesh.clone())).unwrap();
        assert!(dynamic_lowering.stable_hlo().contains("stablehlo.get_dimension_size"));
        assert!(dynamic_lowering.stable_hlo().contains("stablehlo.add"));
        assert!(dynamic_lowering.stable_hlo().contains("stablehlo.broadcast_in_dim"));
        assert!(dynamic_lowering.stable_hlo().contains("stablehlo.set_dimension_size"));
        assert!(dynamic_lowering.stable_hlo().contains("stablehlo.dynamic_reshape"));

        // Persistent compilation identity canonicalizes diagnostic-only variable names while retaining dimension-SSA
        // semantics. Rebuilding the same graph over a fresh nominal identity therefore shares its compilation key,
        // while changing the dimension arithmetic does not.
        let renamed_extent = DimensionVariable::new("renamed_extent", DimensionBounds::new(1, Some(8)).unwrap());
        let renamed_type = ArrayType::new(DataType::F32, Shape::new(vec![renamed_extent.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let renamed_program = build_program(&renamed_type, 1);
        let renamed_lowering = domain.lower_program(&renamed_program, 0, &Options::new(mesh.clone())).unwrap();
        let different_program = build_program(&vector_type, 2);
        let different_lowering = domain.lower_program(&different_program, 0, &Options::new(mesh.clone())).unwrap();
        let dynamic_key = domain.compilation_key(&dynamic_lowering).unwrap();
        assert_eq!(dynamic_key, domain.compilation_key(&renamed_lowering).unwrap());
        assert_ne!(dynamic_key, domain.compilation_key(&different_lowering).unwrap());

        // CPU PJRT does not provide the bounded-input `PadToStatic` custom call. Execute the same first-class
        // dimension graph with a static input axis; this changes only dimension-size lowering from a runtime read to
        // its exact scalar constant and still exercises dimension SSA arithmetic and both mixed shape operations.
        let static_vector_type = replicated_vector_type(&mesh, 3);
        let static_program = build_program(&static_vector_type, 1);
        let static_lowering = domain.lower_program(&static_program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&static_lowering).unwrap();
        let vector = Array::from_host_buffer(
            &domain,
            static_vector_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0]),
        )
        .unwrap();
        let scalar = f32_scalar(&domain, &mesh, 7.0);
        let outputs = domain.execute_program(&compiled, vec![vector, scalar]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape().as_slice(), &[4]);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![7.0; 4]);
    }

    #[test]
    fn test_production_array_ir_lowering_executes_dynamic_slice_with_dimensions() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = replicated_vector_type(&mesh, 6);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let start = builder
            .add_instruction(
                Operation::Dimension(DimensionOperation::Constant(ConstantOperation::new(
                    DimensionValue::constant(1).unwrap(),
                ))),
                Vec::new(),
                Vec::new(),
                None,
            )
            .unwrap()[0];
        let size = builder
            .add_instruction(
                Operation::Dimension(DimensionOperation::Constant(ConstantOperation::new(
                    DimensionValue::constant(3).unwrap(),
                ))),
                Vec::new(),
                Vec::new(),
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(
                Operation::DynamicSlice(
                    DynamicSliceOperation::<ArrayIrType>::from_rank(1).with_strides(vec![2]).unwrap(),
                ),
                Vec::new(),
                vec![input, start, size],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let lowering = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert!(lowering.stable_hlo().contains("stablehlo.dynamic_slice"));
        assert!(lowering.stable_hlo().contains("stablehlo.slice"));
        assert!(!lowering.stable_hlo().contains("stablehlo.real_dynamic_slice"));
        let compiled = domain.compile_program(&lowering).unwrap();
        let input =
            Array::from_host_buffer(&domain, input_type, mesh, values_to_bytes::<f32>(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
                .unwrap();
        let outputs = domain.execute_program(&compiled, vec![input]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape().as_slice(), &[3]);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_compilation_domain_impl_round_trips_through_core_pipeline() {
        use crate::tests::{values_from_bytes, values_to_bytes};
        use ryft_core::Sin;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let engine = Session::new(&client).domain().with_mesh(mesh.clone());

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let options = Options::new(mesh.clone());
        let staged = crate::stage::<_, ArrayType, ArrayType>(|x| x.sin().unwrap(), input_type.clone(), options)
            .unwrap()
            .into_inner();
        let compiled: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
            staged.lower(&engine).unwrap().compile(&engine).unwrap();

        // Round-trip a small input through the new CompilationDomain-driven pipeline.
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let array = compiled.executable().execute(&engine, Value::Array(source)).unwrap();
        let Value::Array(array) = array else {
            panic!("array-only compiled function returned a first-class dimension");
        };
        array.block_until_ready().unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }

        // Equivalent lowerings share the entry populated by the first compilation, independent of source location.
        let program_count_before = engine.program_count();
        for _ in 0..3 {
            let staged = crate::stage::<_, ArrayType, ArrayType>(
                |x| x.sin().unwrap(),
                input_type.clone(),
                Options::new(mesh.clone()),
            )
            .unwrap()
            .into_inner();
            let _: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
                staged.lower(&engine).unwrap().compile(&engine).unwrap();
        }
        assert_eq!(
            engine.program_count(),
            program_count_before,
            "equivalent repeat compilations should reuse the existing cache entry",
        );
    }

    #[test]
    #[allow(deprecated)] // Inspect the actual buffer, independently of the declared array type.
    fn test_lowering_preserves_cast_layouts() {
        use ryft_core::{Cast, CastOperation};

        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let actual_type = ArrayType::new_static(DataType::F64, [2, 3]);
        let values = vec![1f64, 2., 3., 4., 5., 6.];
        let input =
            Array::from_host_buffer(&domain, actual_type.clone(), mesh.clone(), values_to_bytes(&values)).unwrap();
        let target = input.r#type().into_owned().with_layout(layout.clone());
        let consumer = crate::compile::<_, ArrayType, ArrayType>(
            |input| input.clone() + input,
            input.r#type().into_owned(),
            &domain,
            mesh.clone(),
        )
        .unwrap();
        for output in [input.cast(target.clone(), &[]).unwrap(), input.cast_unchecked(target, &[]).unwrap()] {
            assert_eq!(
                output.addressable_shards().next().unwrap().buffer().unwrap().layout(),
                Ok(layout.to_pjrt().unwrap())
            );
            // The next executable must consume the actual column-major buffer and return dense logical values.
            let consumed = output.cast(input.r#type().into_owned(), &[]).unwrap();
            assert_eq!(read_f64s(&client, &consumed), values);
            assert_eq!(values_from_bytes::<f64>(&materialize_dense_array_bytes(&output).unwrap()), values);
            let destination_mesh = domain_mesh(&client, "destination", 1);
            let moved = output
                .to_placement(
                    &domain,
                    crate::arrays_v0::DevicePutTarget::Placement {
                        sharding: Sharding::replicated(destination_mesh.logical_mesh().clone(), 2),
                        mesh: destination_mesh,
                    },
                )
                .unwrap();
            assert_eq!(moved.layout(), Some(&layout));
            assert_eq!(values_from_bytes::<f64>(&materialize_dense_array_bytes(&moved).unwrap()), values);
            // A raw buffer may have a known physical layout even when its logical type leaves that field open.
            let opaque = Array::from_addressable_buffers(
                &domain,
                output.r#type().into_owned().with_layout(None),
                mesh.clone(),
                vec![
                    output
                        .addressable_shards()
                        .next()
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .copy_to_device(client.addressable_devices().unwrap().remove(0))
                        .unwrap(),
                ],
            )
            .unwrap();
            let doubled = consumer.executable().execute(&domain, opaque).unwrap();
            assert_eq!(read_f64s(&client, &doubled), vec![2., 4., 6., 8., 10., 12.]);
            let doubled = consumer.executable().execute(&domain, output).unwrap();
            assert_eq!(read_f64s(&client, &doubled), vec![2., 4., 6., 8., 10., 12.]);
        }

        // A tuple mixes an erased zero, a constrained result and an unconstrained result. The dynamic case also
        // appends hidden extent results. Check the actual buffers rather than trusting their reconstructed types.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        let zero_type = ArrayType::new_static(DataType::Zero, [2, 3]);
        let zero = Array::from_host_buffer(&domain, zero_type.clone(), mesh.clone(), []).unwrap();
        for source_type in [actual_type, ArrayType::new(DataType::F64, Shape::new(vec![extent.into(), 3.into()]))] {
            for constructor in [CastOperation::<ArrayIrType>::checked, CastOperation::<ArrayIrType>::unchecked] {
                let mut builder = ProgramBuilder::new();
                let source = builder.add_input(source_type.clone().into());
                let zero_input = builder.add_input(zero_type.clone().into());
                let output = builder
                    .add_instruction(
                        constructor(
                            input.r#type().into_owned().with_layout(layout.clone()).into(),
                            vec![source_type.clone().into()],
                        )
                        .unwrap(),
                        vec![],
                        vec![source],
                        None,
                    )
                    .unwrap()[0];
                let program = builder
                    .build::<Vec<Constant>, Vec<Constant>>(
                        vec![zero_input, output, source],
                        vec![Placeholder; 2],
                        vec![Placeholder; 3],
                    )
                    .unwrap();
                let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
                assert_eq!(lowered.signature.input_mapping(), &[Some(0), None]);
                assert_eq!(lowered.signature.output_mapping(), &[None, Some(0), Some(1)]);
                let compiled = domain.compile_program(&lowered).unwrap();
                let outputs = domain.execute_program(&compiled, vec![input.clone(), zero.clone()]).unwrap();
                assert_eq!(outputs[1].shape().as_slice(), &[2, 3]);
                assert_eq!(outputs[1].layout(), Some(&layout));
                assert_eq!(
                    outputs[1].addressable_shards().next().unwrap().buffer().unwrap().layout(),
                    Ok(layout.to_pjrt().unwrap())
                );
                assert_eq!(values_from_bytes::<f64>(&materialize_dense_array_bytes(&outputs[1]).unwrap()), values);
                assert_eq!(read_f64s(&client, &outputs[2]), values);
                assert!(outputs[0].addressable_shards().all(|shard| shard.buffer().is_none()));
            }
        }
    }

    #[test]
    #[allow(deprecated)] // Verify actual result storage alongside the hidden extent and effect-token ABI.
    fn test_lowering_preserves_dynamic_row_major_layout_with_ordered_io() {
        use ryft_core::CastOperation;

        use crate::operations::debugging::with_captured_prints;

        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        let source_type = ArrayType::new(DataType::F64, Shape::new(vec![extent.into(), 3.into()]));
        let layout = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(source_type.clone().into());
        let output = builder
            .add_instruction(
                CastOperation::<ArrayIrType>::checked(
                    source_type.clone().with_layout(layout.clone()).into(),
                    vec![source_type.into()],
                )
                .unwrap(),
                vec![],
                vec![input],
                None,
            )
            .unwrap()[0];
        let scalar = builder.add_input(ArrayType::scalar(DataType::F64).into());
        builder
            .add_instruction(PrintOperation::<ArrayType>::new("layout"), vec![], vec![scalar], None)
            .unwrap();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.signature.ordered_io_output_index(), Some(2));
        let compiled = domain.compile_program(&lowered).unwrap();
        for rows in [2, 0] {
            let values = (1..=rows * 3).map(|value| value as f64).collect::<Vec<_>>();
            let input = Array::from_host_buffer(
                &domain,
                ArrayType::new_static(DataType::F64, [rows, 3]),
                mesh.clone(),
                values_to_bytes(&values),
            )
            .unwrap();
            let scalar = Array::from_host_buffer(
                &domain,
                ArrayType::scalar(DataType::F64),
                mesh.clone(),
                values_to_bytes(&[1f64]),
            )
            .unwrap();
            let (outputs, prints) =
                with_captured_prints(|| domain.execute_program(&compiled, vec![input, scalar]).unwrap());
            assert_eq!(outputs[0].shape().as_slice(), &[rows, 3]);
            assert_eq!(
                outputs[0].addressable_shards().next().unwrap().buffer().unwrap().layout(),
                Ok(layout.to_pjrt().unwrap()),
            );
            assert_eq!(values_from_bytes::<f64>(&materialize_dense_array_bytes(&outputs[0]).unwrap()), values);
            assert_eq!(prints.len(), 1);
        }
    }

    #[test]
    #[allow(deprecated)] // Inspect per-device storage in addition to the global type.
    fn test_lowering_preserves_cast_layouts_on_multiple_cpu_devices() {
        use ryft_core::Cast;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        // Types record placement over explicit mesh axes only, so the sharded cast type uses an explicit axis.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            client
                .addressable_devices()
                .unwrap()
                .into_iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect(),
        )
        .unwrap();
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = ArrayType::new_static(DataType::F64, [4, 3])
            .with_sharding(
                Sharding::new(
                    mesh.logical_mesh().clone(),
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
                )
                .unwrap(),
            )
            .unwrap();
        let values = (1..=12).map(|value| value as f64).collect::<Vec<_>>();
        let input =
            Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), values_to_bytes(&values)).unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let target = input_type.with_layout(layout.clone());
        let output = input.cast(target, &[]).unwrap();
        for shard in output.addressable_shards() {
            assert_eq!(shard.shape().as_slice(), &[2, 3]);
            assert_eq!(shard.buffer().unwrap().layout(), Ok(layout.to_pjrt().unwrap()));
        }
        assert_eq!(values_from_bytes::<f64>(&materialize_dense_array_bytes(&output).unwrap()), values);
    }

    #[test]
    fn test_data_derived_compilation_rejects_unbounded_and_opaque_consumers() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);

        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let extent = DimensionVariable::new("unbounded", DimensionBounds::unbounded());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![dimension], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            domain.lower_program(&program, 0, &Options::new(mesh.clone())),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::InvalidArgument { message })
                    if *message == "data-derived dimension `unbounded` with bounds [0, ∞) needs a finite upper bound \
                        for XLA compilation",
            ),
        ));

        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let input = builder
            .add_instruction(IotaOperation::new(dynamic_type.clone(), 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(
                CustomCallOperation::new("ryft.test.opaque_dynamic", vec![dynamic_type]),
                Vec::new(),
                vec![input, dimension],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            domain.lower_program(&program, 0, &Options::new(mesh.clone())),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message == "operation `custom_call` cannot consume data-derived dimensions during XLA lowering \
                        because custom-call physical-padding semantics are opaque",
            ),
        ));

        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let input = builder
            .add_instruction(IotaOperation::new(dynamic_type, 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Mean), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            domain.lower_program(&program, 0, &Options::new(mesh.clone())),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message == "operation `reduce` cannot consume data-derived dimensions during XLA lowering \
                        because mean over a dynamically sized reduction axis is not supported by the XLA lowering",
            ),
        ));

        // `print` renders whatever the physical buffer holds, so the padded suffix of a bounded-dynamic operand would
        // become observable.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let input = builder
            .add_instruction(IotaOperation::new(dynamic_type, 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(PrintOperation::<ArrayType>::new("value"), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            domain.lower_program(&program, 0, &Options::new(mesh.clone())),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message
                        == "operation `print` cannot consume data-derived \
                        dimensions during XLA lowering because \
                        print exposes the physical representation of its operand",
            ),
        ));

        // Bit generation over a bounded-dynamic result would advance the functional generator state by the physical
        // upper-bound element count rather than the logical one.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let state = builder.add_input(RandomAlgorithm::ThreeFry.state_type().into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(
                RngBitGeneratorOperation::<ArrayIrType>::new(RandomAlgorithm::ThreeFry, bits_type),
                Vec::new(),
                vec![state, dimension],
                None,
            )
            .unwrap()[1];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            domain.lower_program(&program, 0, &Options::new(mesh.clone())),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message == "operation `rng_bit_generator` cannot consume data-derived dimensions during XLA \
                        lowering because random generation advances state according to physical rather than logical \
                        element count",
            ),
        ));

        // Every scatter combiner is unverified against physical padding, so a data-derived operand extent is rejected
        // per kind rather than per operation name.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let indices_type =
            ArrayType::new(DataType::I32, Shape::new(vec![CoreDimension::Static(2), CoreDimension::Static(1)]));
        let updates_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]));
        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.into());
        let indices = builder.add_input(indices_type.into());
        let updates = builder.add_input(updates_type.into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let operand = builder
            .add_instruction(IotaOperation::new(dynamic_type, 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(
                ScatterOperation::new(
                    ScatterDimensionNumbers::new(Vec::new(), vec![0], vec![0]),
                    ScatterReductionKind::Add,
                ),
                Vec::new(),
                vec![operand, indices, updates],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            domain.lower_program(&program, 0, &Options::new(mesh)),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message
                        == "operation `scatter` cannot consume data-derived \
                        dimensions during XLA lowering because \
                        add-scatter padding semantics have not been verified for data-derived dimensions",
            ),
        ));
    }

    #[test]
    fn test_input_bound_bucketing_reuses_power_of_two_specializations() {
        fn trace<'c>(input: CompilationTracer<Domain<'c>>) -> Result<CompilationTracer<Domain<'c>>, Error> {
            let context = input.context().clone();
            Ok(context.bind(AddOperation::new(), Vec::new(), &[input.clone(), input])?.remove(0))
        }

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let function: CompiledFunctionDispatcher<Domain<'_>, (), ArrayIrType, ArrayIrType, _> = domain
            .jit_with_options(
                |(), input| trace(input),
                Options::new(mesh.clone()).with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo),
            );
        let call = |values: &[f32]| {
            let input = f32_vector(&domain, &mesh, values);
            function.execute((), Value::Array(input)).unwrap()
        };

        let three = call(&[1.0, 2.0, 3.0]);
        assert_eq!(program_array(&three).shape(), StaticShape::new(vec![3]));
        assert_eq!(read_f32s(&client, program_array(&three)), vec![2.0, 4.0, 6.0]);
        let four = call(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(program_array(&four).shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, program_array(&four)), vec![2.0, 4.0, 6.0, 8.0]);
        let five = call(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(program_array(&five).shape(), StaticShape::new(vec![5]));
        assert_eq!(read_f32s(&client, program_array(&five)), vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.dispatch_misses, 2);
        assert_eq!(statistics.traces, 2);
        assert_eq!(statistics.lowerings, 2);
        assert_eq!(statistics.compilation_requests, 2);
        assert_eq!(function.executable_specialization_count(), 2);
        assert_eq!(domain.program_count(), 2);
    }

    #[test]
    fn test_domain_dispatch_signature() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let input = Value::Array(
            Array::from_host_buffer(
                &domain,
                ArrayType::new_static(DataType::F32, [3]),
                mesh.clone(),
                values_to_bytes(&[1f32, 2.0, 3.0]),
            )
            .unwrap(),
        );
        let input_types = vec![input.r#type().into_owned()];
        let options = Options::new(mesh).with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo);

        let (key, staged_types, specialized_options) =
            Domain::dispatch_signature(DispatchInputs::Types(input_types.clone()), &options).unwrap();
        assert_eq!(key, DispatchKey::new(DispatchKeyKind::Exact(input_types.clone().into())));
        assert_eq!(staged_types.as_ref(), input_types.as_slice());
        assert_eq!(specialized_options, None);

        let (key, executable_types, specialized_options) = Domain::dispatch_signature(
            DispatchInputs::Values { values: &mut std::iter::once(&input), types: input_types.clone() },
            &options,
        )
        .unwrap();
        assert_eq!(key, normalize_dispatch_signature(input_types, &options, options.input_bound_bucketing).unwrap().0);
        let [ArrayIrType::Array(array_type)] = executable_types.as_ref() else {
            panic!("the executable signature must contain one array");
        };
        let [CoreDimension::Dynamic(extent)] = array_type.shape().dimensions() else {
            panic!("runtime values must select input-bound bucketing");
        };
        assert_eq!(extent.name(), "input_0_axis_0");
        assert_eq!(extent.bounds(), DimensionBounds::new(0, Some(5)).unwrap());
        assert_eq!(specialized_options, None);
    }

    #[test]
    fn test_domain_dispatch_signature_runtime_placement() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 2);
        let domain = Session::new(&client).domain();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let tiled = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated_input = Value::Array(
            Array::from_host_buffer_with_sharding(
                &domain,
                ArrayType::new_static(DataType::F32, [4]),
                replicated.clone(),
                mesh.clone(),
                values_to_bytes(&[1f32, 2.0, 3.0, 4.0]),
            )
            .unwrap(),
        );
        let tiled_input = Value::Array(
            Array::from_host_buffer_with_sharding(
                &domain,
                ArrayType::new_static(DataType::F32, [4]),
                tiled.clone(),
                mesh.clone(),
                values_to_bytes(&[1f32, 2.0, 3.0, 4.0]),
            )
            .unwrap(),
        );
        assert_eq!(replicated_input.r#type(), tiled_input.r#type());
        let input_types = vec![replicated_input.r#type().into_owned()];
        let options = Options::new(mesh);
        let (replicated_key, replicated_types, replicated_options) = Domain::dispatch_signature(
            DispatchInputs::Values { values: &mut std::iter::once(&replicated_input), types: input_types.clone() },
            &options,
        )
        .unwrap();
        let (tiled_key, tiled_types, tiled_options) = Domain::dispatch_signature(
            DispatchInputs::Values { values: &mut std::iter::once(&tiled_input), types: input_types.clone() },
            &options,
        )
        .unwrap();
        assert_ne!(replicated_key, tiled_key);
        assert_eq!(replicated_types, tiled_types);
        assert_eq!(replicated_options, None);
        assert_eq!(
            tiled_options,
            Some(Options { input_placements: Some(vec![Some(tiled)].into()), ..options.clone() }),
        );

        // A declared sharding determines placement instead of the actual placement of either input.
        let options = options.with_input_shardings(vec![replicated]);
        let replicated_signature = Domain::dispatch_signature(
            DispatchInputs::Values { values: &mut std::iter::once(&replicated_input), types: input_types.clone() },
            &options,
        )
        .unwrap();
        let tiled_signature = Domain::dispatch_signature(
            DispatchInputs::Values { values: &mut std::iter::once(&tiled_input), types: input_types },
            &options,
        )
        .unwrap();
        assert_eq!(replicated_signature, tiled_signature);
        assert_eq!(replicated_signature.0, replicated_key);
        assert_eq!(replicated_signature.2, None);
    }

    #[test]
    fn test_input_bound_bucketing_rejects_unsupported_signatures() {
        let plugin = load_cpu_plugin().unwrap();
        let single_device_client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let single_device_mesh = domain_mesh(&single_device_client, "x", 1);
        let single_device_domain = Session::new(&single_device_client).domain().with_mesh(single_device_mesh.clone());
        let dimension_type = DimensionType::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        assert!(matches!(
            normalize_dispatch_signature(vec![dimension_type.clone().into()], &Options::new(single_device_mesh.clone())
                    .with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo), Some(InputBoundBucketing::PowerOfTwo)),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == format!(
                    "input-bound bucketing only supports array inputs, but input 0 has first-class dimension type {}",
                    dimension_type,
                ),
        ));
        // Exact dispatch admits structural reference types for the opt-in stateful call surface, while bucketed
        // dispatch rejects them because holder state is not part of the array-only bucketing model.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let reference_input_types: Arc<[ArrayIrType]> = vec![ArrayIrType::Reference(reference_type.clone())].into();
        assert_eq!(
            normalize_dispatch_signature(
                reference_input_types.to_vec(),
                &Options::new(single_device_mesh.clone()),
                None
            )
            .unwrap(),
            (DispatchKey::new(DispatchKeyKind::Exact(reference_input_types.clone())), reference_input_types,),
        );
        assert!(matches!(
            normalize_dispatch_signature(vec![ArrayIrType::Reference(reference_type.clone())], &Options::new(single_device_mesh.clone())
                    .with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo), Some(InputBoundBucketing::PowerOfTwo)),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference input 0 does not support input-bound bucketing",
        ));
        assert!(matches!(
            single_device_domain.lower_program(
                &external_reference_read_program(ArrayType::scalar(DataType::F32)),
                1,
                &Options::new(single_device_mesh.clone())
                    .with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo),
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference inputs do not support input-bound bucketing",
        ));

        let multi_device_client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let multi_device_mesh = domain_mesh(&multi_device_client, "x", 2);
        assert!(matches!(
            normalize_dispatch_signature(vec![ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(3)])).into()], &Options::new(multi_device_mesh).with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo), Some(InputBoundBucketing::PowerOfTwo)),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "input-bound bucketing is unsupported on multi-device meshes because bucketed dynamic \
                    signatures require disabling the Shardy partitioner",
        ));

        // Multi-device bucketing is rejected before inspecting individual input kinds.
        assert!(matches!(
            normalize_dispatch_signature(vec![ArrayIrType::Reference(reference_type.clone())], &Options::new(domain_mesh(&multi_device_client, "x", 2))
                    .with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo), Some(InputBoundBucketing::PowerOfTwo)),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "input-bound bucketing is unsupported on multi-device meshes because bucketed dynamic \
                    signatures require disabling the Shardy partitioner",
        ));
    }

    #[test]
    fn test_input_bound_bucketing_alpha_normalizes_dispatch_keys() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let options = Options::new(mesh).with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo);
        let bounds = DimensionBounds::new(0, Some(5)).unwrap();
        let vector_type =
            |dimension: CoreDimension| ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![dimension])));
        let key = |input_types: Vec<ArrayIrType>| {
            normalize_dispatch_signature(input_types, &options, options.input_bound_bucketing).unwrap().0
        };

        // Two runtime signatures landing in the same power-of-two bucket share one key, and the effective staged
        // types carry the generated per-axis variable names that the key itself excludes.
        let (three_key, three_types) = normalize_dispatch_signature(
            vec![vector_type(CoreDimension::Static(3))],
            &options,
            options.input_bound_bucketing,
        )
        .unwrap();
        assert_eq!(three_key, key(vec![vector_type(CoreDimension::Static(4))]));
        assert_ne!(three_key, key(vec![vector_type(CoreDimension::Static(5))]));
        let [ArrayIrType::Array(three_type)] = three_types.as_ref() else {
            panic!("bucketing must stage exactly one array input type");
        };
        let [CoreDimension::Dynamic(variable)] = three_type.shape().dimensions() else {
            panic!("bucketing must stage a rank-one bounded dynamic axis");
        };
        assert_eq!(variable.name(), "input_0_axis_0");
        assert_eq!(variable.bounds(), bounds);

        // Signatures requested for staging preserve the caller's types without executable dispatch bucketing.
        let (_, staged_types, _) =
            Domain::dispatch_signature(DispatchInputs::Types(vec![vector_type(CoreDimension::Static(3))]), &options)
                .unwrap();
        assert_eq!(staged_types.as_ref(), &[vector_type(CoreDimension::Static(3))]);

        // Dimension variables are keyed by their bounds and first-occurrence sharing pattern, not by their names, so
        // renaming a variable preserves the key while resharing it across inputs does not.
        let first = DimensionVariable::new("first", bounds);
        let second = DimensionVariable::new("second", bounds);
        let renamed = DimensionVariable::new("renamed", bounds);
        assert_eq!(
            key(vec![vector_type(first.clone().into()), vector_type(first.clone().into())]),
            key(vec![vector_type(renamed.clone().into()), vector_type(renamed.into())]),
        );
        assert_ne!(
            key(vec![vector_type(first.clone().into()), vector_type(first.clone().into())]),
            key(vec![vector_type(first.clone().into()), vector_type(second.into())]),
        );

        // Everything the shape-free carrier retains still discriminates: element data type, and here the rank too.
        assert_ne!(
            key(vec![vector_type(first.clone().into())]),
            key(vec![ArrayType::new(DataType::F64, Shape::new(vec![first.clone().into()])).into()]),
        );
        assert_ne!(
            key(vec![vector_type(first.clone().into())]),
            key(vec![ArrayType::new(DataType::F32, Shape::new(vec![first.clone().into(), first.into()])).into()]),
        );
    }

    #[test]
    fn test_bounded_dynamic_programs_reject_shard_map_regions_before_device_count_matters() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Manual).unwrap()]).unwrap();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let value_type = replicated_scalar_type(&mesh, DataType::F32);
        let global_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let manual_sharding = Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let boundary =
            ShardMapBoundary::new(manual_mesh, vec![manual_sharding.clone()], vec![manual_sharding], Vec::new())
                .unwrap();
        let local_type = boundary.local_input_type(0, &global_type).unwrap();

        // A `shard_map` body always has static boundary types, so it can coexist with dynamic tensors that are staged
        // beside it. Here the gateway makes the program bounded-dynamic while the `shard_map` stays fully static.
        let mut body_builder = ProgramBuilder::new();
        let local = body_builder.add_input(local_type.into());
        let doubled =
            body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![local, local], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let size = builder.add_input(size_type.into());
        let value = builder.add_input(value_type.into());
        let global = builder.add_input(global_type.clone().into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let dynamic = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![value, dimension], None)
            .unwrap()[0];
        let operation = ShardMapOperation::from_program(&body, vec![global_type.into()], boundary).unwrap();
        let mapped = builder
            .add_instruction(Operation::ShardMap(Box::new(operation)), vec![body_region], vec![global], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![mapped, dynamic],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();

        let Err(error) = domain.lower_program(&program, 0, &Options::new(mesh)) else {
            panic!("bounded-dynamic program containing a shard_map unexpectedly lowered");
        };
        assert!(matches!(&error,
            Error::Lowering(crate::LoweringError::UnsupportedOperation { operation, .. }) if operation == "shard_map",
        ));
        assert_eq!(
            error.to_string(),
            "cannot lower `shard_map`: bounded-dynamic programs cannot contain shard_map regions because \
             sdy.manual_computation requires the Shardy partitioner",
        );
    }

    #[test]
    fn test_compilation_key_is_canonical_and_stable() {
        use ryft_core::Sin;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(|x| x.sin().unwrap(), input_type, Options::new(mesh))
            .unwrap()
            .into_inner();
        let compiled: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
            staged.lower(&domain).unwrap().compile(&domain).unwrap();
        let key = domain.compilation_key(compiled.lowered().lowered_program()).unwrap();
        let independent_domain = Session::new(&client).domain();
        let repeated = independent_domain.compilation_key(compiled.lowered().lowered_program()).unwrap();

        assert_eq!(key, repeated);
        assert_eq!(domain.persistent_cache_key(&key), Some(key.canonical_bytes.to_vec()));
        let decoded: serde_json::Value = serde_json::from_slice(&key.canonical_bytes).unwrap();
        assert_eq!(decoded["schema_version"], PERSISTENT_KEY_SCHEMA_VERSION);
        assert_eq!(decoded["runtime_requirements"], serde_json::json!([]));
        assert_eq!(decoded["compiler_identity"], COMPILER_IDENTITY.as_str());
        assert!(decoded["compiler_identity"].as_str().unwrap().contains(ryft_xla_sys::XLA_COMMIT));
        assert!(decoded["compiler_identity"].as_str().unwrap().contains(ryft_xla_sys::JAX_COMMIT));
        assert_eq!(decoded["platform_name"], client.platform_name().unwrap().as_ref());
        assert!(decoded["compilation_options"].as_array().is_some_and(|bytes| !bytes.is_empty()));
        assert!(decoded["stable_hlo"].as_str().is_some_and(|module| module.contains("stablehlo.sine")));
    }

    #[test]
    fn test_compilation_key_includes_runtime_abi_and_configuration() {
        use crate::domains::runtimes::{RuntimeRequirement, with_runtime_requirements};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let array_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let lower = |requirements| {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(array_type.clone().into());
            let operation = with_runtime_requirements(
                CustomCallOperation::new("ryft.test.runtime", vec![array_type.clone()]),
                requirements,
            )
            .unwrap();
            let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap()
        };
        let requirement =
            |version, configuration| RuntimeRequirement::new("test.runtime", version, configuration).unwrap();
        let first = lower(vec![requirement(1, vec![7])]);
        let duplicate = lower(vec![requirement(1, vec![7]), requirement(1, vec![7])]);
        let changed_abi = lower(vec![requirement(2, vec![7])]);
        let changed_configuration = lower(vec![requirement(1, vec![8])]);

        assert_eq!(first.stable_hlo(), changed_abi.stable_hlo());
        assert_eq!(first.stable_hlo(), changed_configuration.stable_hlo());
        assert_eq!(domain.compilation_key(&first).unwrap(), domain.compilation_key(&duplicate).unwrap());
        assert_ne!(domain.compilation_key(&first).unwrap(), domain.compilation_key(&changed_abi).unwrap());
        assert_ne!(domain.compilation_key(&first).unwrap(), domain.compilation_key(&changed_configuration).unwrap());
    }

    #[test]
    fn test_stage_enforces_reference_boundary_sharding_policy() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_sharding = Sharding::new(mesh.logical_mesh().clone(), Vec::new()).unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));

        // Reference inputs are structurally valid for stateful staging, but user-provided input sharding overrides
        // are not: holder state derives its effective sharding from the compiled reference ABI. Lowering rejects the
        // overrides for staged functions whose input types did not receive them.
        fn forward<'c>(
            _: Vec<Constant>,
            _: Vec<CompilationTracer<Domain<'c>>>,
            input: CompilationTracer<Domain<'c>>,
        ) -> Result<CompilationTracer<Domain<'c>>, Error> {
            Ok(input)
        }
        assert!(matches!(
            CoreStagedFunction::<Domain<'_>, ArrayIrType, ArrayIrType>::stage(forward,
                Vec::new(),
                reference_type.clone(),
                Options::new(mesh.clone()).with_input_shardings(vec![scalar_sharding.clone()]),
            )
            .and_then(|staged| staged.lower(&domain))
            .map(|_| ()),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference inputs do not support explicit input shardings",
        ));

        // A reference output with `output_shardings` stages successfully and is rejected when lowered, by the
        // reference-output contract of the ABI rather than by the output override, which never runs on a reference
        // type.
        fn allocate<'c>(
            _: Vec<Constant>,
            _: Vec<CompilationTracer<Domain<'c>>>,
            input: CompilationTracer<Domain<'c>>,
        ) -> Result<CompilationTracer<Domain<'c>>, Error> {
            Ok(input.reference_new()?)
        }
        assert!(matches!(
            CoreStagedFunction::<Domain<'_>, ArrayIrType, ArrayIrType>::stage(allocate,
                Vec::new(),
                ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                Options::new(mesh.clone()).with_output_shardings(vec![scalar_sharding]),
            )
            .and_then(|staged| staged.lower(&domain))
            .map(|_| ()),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "output 0 is a reference allocated inside the program and cannot be returned through \
                              the stateful XLA ABI, which has no protocol to turn a backend array into a host \
                              reference",
        ));

        // Forwarding an external reference through an output is rejected by the same contract when lowered, because
        // the ABI has no protocol for handing an entering reference back through an output either.
        assert!(matches!(
            CoreStagedFunction::<Domain<'_>, ArrayIrType, ArrayIrType>::stage(forward,
                Vec::new(),
                reference_type,
                Options::new(mesh.clone()),
            )
            .and_then(|staged| staged.lower(&domain))
            .map(|_| ()),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "output 0 forwards the external reference bound at input 0, and the stateful XLA ABI \
                              does not return forwarded references; read the reference after the call instead",
        ));
    }

    #[test]
    fn test_lower_requires_staged_input_types_to_carry_input_shardings() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let options = Options::new(mesh.clone())
            .with_input_shardings(vec![Sharding::new(mesh.logical_mesh().clone(), Vec::new()).unwrap()]);

        fn forward<'c>(
            _: Vec<Constant>,
            _: Vec<CompilationTracer<Domain<'c>>>,
            input: CompilationTracer<Domain<'c>>,
        ) -> Result<CompilationTracer<Domain<'c>>, Error> {
            Ok(input)
        }

        // Staging traces exactly the input types that it receives, so lowering rejects a function whose input types
        // did not receive the requested input shardings before staging.
        let staged = CoreStagedFunction::<Domain<'_>, ArrayIrType, ArrayIrType>::stage(
            forward,
            Vec::new(),
            scalar_type.clone(),
            options.clone(),
        )
        .unwrap();
        assert!(matches!(
            staged.lower(&domain).map(|_| ()),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "staged input types do not carry the shardings requested by `input_shardings`; apply them \
                              to the input types before staging (e.g., with `with_input_shardings`)",
        ));

        let input_types = with_input_shardings(scalar_type, &options).unwrap();
        let staged = CoreStagedFunction::<Domain<'_>, ArrayIrType, ArrayIrType>::stage(
            forward,
            Vec::new(),
            input_types,
            options,
        )
        .unwrap();
        assert!(staged.lower(&domain).is_ok());
    }

    #[test]
    fn test_lower_requires_staged_output_types_to_carry_output_shardings() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let mesh = |axis_type| {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let devices = client
                .addressable_devices()
                .unwrap()
                .into_iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect::<Vec<_>>();
            DeviceMesh::new(logical_mesh, devices).unwrap()
        };
        let input_type = |mesh: &DeviceMesh| {
            let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated).unwrap())
        };
        let options = |mesh: &DeviceMesh| {
            let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            Options::new(mesh.clone()).with_output_shardings(vec![sharded])
        };

        fn forward<'c>(
            _: Vec<Constant>,
            _: Vec<CompilationTracer<Domain<'c>>>,
            input: CompilationTracer<Domain<'c>>,
        ) -> Result<CompilationTracer<Domain<'c>>, Error> {
            Ok(input)
        }

        fn reshard<'c>(
            _: Vec<Constant>,
            _: Vec<CompilationTracer<Domain<'c>>>,
            input: CompilationTracer<Domain<'c>>,
        ) -> Result<CompilationTracer<Domain<'c>>, Error> {
            let mesh = <&ArrayType>::try_from(input.r#type().as_ref()).unwrap().sharding().unwrap().mesh().clone();
            Ok(input.reshard(&Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())?)
        }

        // Lowering checks rather than overwrites the placement that staged types track, so that calls of a staged
        // function return the output types of its executions.
        let mesh_explicit = mesh(MeshAxisType::Explicit);
        let staged = CoreStagedFunction::<Domain<'_>, ArrayIrType, ArrayIrType>::stage(
            forward,
            Vec::new(),
            input_type(&mesh_explicit),
            options(&mesh_explicit),
        )
        .unwrap();
        assert!(matches!(
            staged.lower(&domain).map(|_| ()),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "staged output 0 has type `f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}]`, which does \
                              not carry the sharding `{mesh<['x'=2:explicit]>, [{'x'}]}` requested by \
                              `output_shardings`; reshard the outputs to their overrides before staging them",
        ));
        let staged = CoreStagedFunction::<Domain<'_>, ArrayIrType, ArrayIrType>::stage(
            reshard,
            Vec::new(),
            input_type(&mesh_explicit),
            options(&mesh_explicit),
        )
        .unwrap();
        assert!(staged.lower(&domain).is_ok());

        // Staged types cannot record placement over auto mesh axes, so lowering applies it to the executable outputs,
        // whose types keep the staged placement while their runtime placements carry the override.
        let mesh_auto = mesh(MeshAxisType::Auto);
        let staged = CoreStagedFunction::<Domain<'_>, ArrayIrType, ArrayIrType>::stage(
            forward,
            Vec::new(),
            input_type(&mesh_auto),
            options(&mesh_auto),
        )
        .unwrap();
        let lowered = staged.lower(&domain).unwrap();
        assert_eq!(
            lowered.lowered_program().output_types()[0].sharding().unwrap().to_string(),
            "{mesh<['x'=2:auto]>, [{}]}",
        );
        assert_eq!(lowered.lowered_program().output_shardings[0].to_string(), "{mesh<['x'=2:auto]>, [{'x'}]}");
    }

    #[test]
    fn test_lowering_rejects_unsupported_external_reference_storage_classes() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let replicated = |rank| Sharding::replicated(mesh.logical_mesh().clone(), rank);

        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(replicated(1))
            .unwrap();
        assert!(matches!(
            domain.lower_program(
                &external_reference_read_program(zero_type),
                0,
                &Options::new(mesh.clone()),
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference state at logical input 0 cannot use a zero-space type because it has \
                              no executable buffer",
        ));

        let host_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(replicated(1))
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        assert!(matches!(
            domain.lower_program(
                &external_reference_read_program(host_type),
                0,
                &Options::new(mesh.clone()),
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference state at logical input 0 must use device memory",
        ));

        let unbounded = DimensionVariable::new("unbounded", DimensionBounds::unbounded());
        let unbounded_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(unbounded)]))
            .with_sharding(replicated(1))
            .unwrap();
        assert!(matches!(
            domain.lower_program(
                &external_reference_read_program(unbounded_type),
                0,
                &Options::new(mesh.clone()),
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference state at logical input 0 has an unbounded dynamic dimension",
        ));

        // Types record placement over explicit mesh axes only, so the sharded state type uses an explicit axis.
        let explicit_logical_mesh =
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let explicit_mesh = DeviceMesh::new(explicit_logical_mesh.clone(), mesh.devices().to_vec()).unwrap();
        let bounded = DimensionVariable::new("bounded", DimensionBounds::new(1, Some(5)).unwrap());
        let sharded_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(bounded)]))
            .with_sharding(Sharding::new(explicit_logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert!(matches!(
            domain.lower_program(&external_reference_read_program(sharded_type), 0, &Options::new(explicit_mesh)),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "bounded-dynamic external state input 0 must use replicated sharding",
        ));

        let foreign_logical_mesh =
            LogicalMesh::new(vec![MeshAxis::new("foreign", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let foreign_sharding_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(foreign_logical_mesh.clone(), 0))
            .unwrap();
        assert!(matches!(
            domain.lower_program(
                &external_reference_read_program(foreign_sharding_type),
                0,
                &Options::new(mesh.clone()),
            ),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Sharding(ryft_core::ShardingError::MeshMismatch { expected, actual })
                    if expected == mesh.logical_mesh() && actual == &foreign_logical_mesh,
            ),
        ));

        let local_device = &mesh.devices()[0];
        let foreign_mesh = DeviceMesh::new(
            mesh.logical_mesh().clone(),
            vec![Device::new(local_device.id(), local_device.process_index() + 1)],
        )
        .unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(foreign_mesh.logical_mesh().clone(), 0))
            .unwrap();
        assert!(matches!(
            domain.lower_program(
                &external_reference_read_program(scalar_type),
                0,
                &Options::new(foreign_mesh),
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference state requires a mesh of devices that this client sees under the processes \
                     that own them, at least one of which this process addresses",
        ));

        let non_addressable_mesh = DeviceMesh::new(
            mesh.logical_mesh().clone(),
            vec![Device::new(local_device.id() + 1_000, local_device.process_index())],
        )
        .unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(non_addressable_mesh.logical_mesh().clone(), 0))
            .unwrap();
        assert!(matches!(
            domain.lower_program(
                &external_reference_read_program(scalar_type),
                0,
                &Options::new(non_addressable_mesh),
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference state requires a mesh of devices that this client sees under the processes \
                     that own them, at least one of which this process addresses",
        ));
    }

    #[test]
    fn test_lowering_discharges_and_executes_local_reference_state() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(scalar_type.clone().into());
        let replacement = builder.add_input(scalar_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, input], None)
            .unwrap();
        let final_value =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program: FlatProgram = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![snapshot, final_value],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.output_types(), &[scalar_type.clone(), scalar_type]);
        let expected_signature = concat!(
            "  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "%arg1: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> ",
            "(tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) {",
        );
        let expected_stable_hlo = indoc! {r#"
            module {
              sdy.mesh @mesh = <["x"=1]>
            @SIGNATURE@
                %0 = stablehlo.add %arg1, %arg0 : tensor<f32>
                return %arg1, %0 : tensor<f32>, tensor<f32>
              }
            }
        "#}
        .replace("@SIGNATURE@", expected_signature);
        assert_eq!(lowered.stable_hlo(), expected_stable_hlo,);
        let compiled = domain.compile_program(&lowered).unwrap();
        let outputs = domain
            .execute_program(&compiled, vec![f32_scalar(&domain, &mesh, 3.0), f32_scalar(&domain, &mesh, 5.0)])
            .unwrap();
        assert_eq!(read_f32s(&client, &outputs[0]), vec![5.0]);
        assert_eq!(read_f32s(&client, &outputs[1]), vec![8.0]);

        let eager_input = ryft_core::ArrayIrValue::Array(CoreArray::scalar(3.0f32).unwrap());
        let eager_replacement = ryft_core::ArrayIrValue::Array(CoreArray::scalar(5.0f32).unwrap());
        let eager_reference = eager_input.reference_new().unwrap();
        eager_reference.write(&eager_replacement).unwrap();
        let eager_snapshot = eager_reference.read().unwrap();
        eager_reference.add_update(&eager_input).unwrap();
        let eager_final = eager_reference.freeze().unwrap();
        assert_eq!(eager_snapshot, ryft_core::ArrayIrValue::Array(CoreArray::scalar(5.0f32).unwrap()));
        assert_eq!(eager_final, ryft_core::ArrayIrValue::Array(CoreArray::scalar(8.0f32).unwrap()));
    }

    #[test]
    fn test_lowering_discharges_and_executes_pinned_host_reference_state_in_shard_maps() {
        // A pinned-host allocation that a `shard_map` mutates through a sharded reference input discharges to a manual
        // computation over its state: every device updates its own shard, the final state keeps the allocation's
        // pinned-host placement, and the mapped output stays in device memory.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let domain = Session::new(&client).domain();
        let sharding = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let device_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let host_type = device_type.clone().with_memory(Memory::Host { pinned: true });
        let boundary = ShardMapBoundary::new(
            logical_mesh,
            vec![sharding.clone(), sharding.clone()],
            vec![sharding],
            vec!["x".to_string()],
        )
        .unwrap();
        let local_type = boundary.local_input_type(1, &device_type).unwrap();
        let body = {
            let mut builder = ProgramBuilder::new();
            let reference = builder.add_input(ReferenceType::new(local_type.clone()).into());
            let update = builder.add_input(local_type.into());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![state], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let operation = ShardMapOperation::from_program(
            &body,
            vec![ReferenceType::new(host_type.clone()).into(), device_type.clone().into()],
            boundary,
        )
        .unwrap();
        let program = {
            let mut builder = ProgramBuilder::new();
            let initial = builder.add_input(host_type.clone().into());
            let update = builder.add_input(device_type.clone().into());
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
            let body = builder.import_program(body);
            let mapped = builder
                .add_instruction(Operation::ShardMap(Box::new(operation)), vec![body], vec![reference, update], None)
                .unwrap()[0];
            let state =
                builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![mapped, state], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let initial = Array::from_host_buffer(
            &domain,
            host_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let update = Array::from_host_buffer(
            &domain,
            device_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[10.0, 20.0, 30.0, 40.0]).as_slice(),
        )
        .unwrap();
        let outputs = domain.execute_program(&compiled, vec![initial, update]).unwrap();
        let shards = |array: &Array<'_>| {
            array
                .addressable_shards()
                .map(|shard| {
                    let buffer = shard.buffer().unwrap();
                    let bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
                    (buffer.memory().unwrap().kind().unwrap().to_string(), values_from_bytes::<f32>(bytes.as_slice()))
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(outputs[0].r#type().as_ref(), &device_type);
        assert_eq!(
            shards(&outputs[0]),
            vec![("device".to_string(), vec![11.0, 22.0]), ("device".to_string(), vec![33.0, 44.0])],
        );
        assert_eq!(outputs[1].r#type().as_ref(), &host_type);
        assert_eq!(
            shards(&outputs[1]),
            vec![("pinned_host".to_string(), vec![11.0, 22.0]), ("pinned_host".to_string(), vec![33.0, 44.0])],
        );
    }

    #[test]
    fn test_lowering_executes_gradients_through_body_allocated_reference_residuals_of_shard_maps() {
        use ryft_core::{Cos, Differentiate, ParallelReduce, Reduce, Sin, composite};

        // The body allocates `r = reference_new(x)`, calls a custom `sin` whose reverse-mode rule saves `r` as a
        // residual and whose backward rule scales the cotangent by the state that it reads from `r`, and then writes
        // `x * x` into `r`. The reference crosses from the primal `shard_map` to the tangent `shard_map` as a snapshot
        // of its final state, from which the tangent body allocates its own reference, so the gradient of
        // `sum(sin(x))` is `cos(x) * x * x`, as it is outside `shard_map`. Both bodies discharge their allocations.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let domain = Session::new(&client).domain();
        let replicated = Sharding::replicated(logical_mesh.clone(), 0);
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        type BodyTracer = Tracer<'static>;
        let function = composite(|(_, x): (BodyTracer, BodyTracer)| {
            Ok(ValueProjection::<ArrayType>::into_projected(x)?.sin()?.into_value())
        })
        .with_non_differentiated_count(1)
        .with_vjp(
            |(r, x): (BodyTracer, BodyTracer)| {
                let x = ValueProjection::<ArrayType>::into_projected(x)?;
                Ok((x.sin()?.into_value(), (r, x.cos()?.into_value())))
            },
            |(r, cosine): (BodyTracer, BodyTracer), cotangent: BodyTracer| {
                let state = ValueProjection::<ArrayType>::into_projected(r.read()?)?;
                let cosine = ValueProjection::<ArrayType>::into_projected(cosine)?;
                let cotangent = ValueProjection::<ArrayType>::into_projected(cotangent)?;
                Ok((r, (cosine * cotangent * state).into_value()))
            },
        );
        let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
                let (value, gradient) = inputs[0]
                    .domain()
                    .differentiate_at(inputs[0].clone())
                    .value_and_gradient(|x| {
                        let x = ValueProjection::<ArrayType>::into_projected(x)?;
                        Ok(shard_map(
                            |local_x: ArrayTracer| {
                                let value = local_x.clone().into_value();
                                let reference = value.reference_new()?;
                                let output = function.call((reference.clone(), value))?;
                                reference.write(&(local_x.clone() * local_x).into_value())?;
                                ValueProjection::<ArrayType>::into_projected(output)?
                                    .reduce(&[0], ReductionKind::Sum)?
                                    .parallel_reduce(ReductionKind::Sum, "x")
                            },
                            x,
                            logical_mesh.clone(),
                            sharded.clone(),
                            replicated.clone(),
                            ShardMapOptions::new(),
                        )
                        .unwrap()
                        .into_value())
                    })
                    .unwrap();
                Ok(vec![value, gradient])
            },
            vec![ArrayIrType::Array(input_type.clone())],
        )
        .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let inputs = [0.25f32, 0.5, 1.0, 2.0];
        let input =
            Array::from_host_buffer(&domain, input_type, mesh, values_to_bytes::<f32>(&inputs).as_slice()).unwrap();
        let outputs = domain.execute_program(&compiled, vec![input]).unwrap();
        let shards = |array: &Array<'_>| {
            array
                .addressable_shards()
                .map(|shard| {
                    let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                    values_from_bytes::<f32>(bytes.as_slice())
                })
                .collect::<Vec<_>>()
        };
        let value = inputs.iter().map(|&input| f32::sin(input)).sum::<f32>();
        for shard in shards(&outputs[0]) {
            assert!((shard[0] - value).abs() <= 1e-5, "{shard:?}");
        }
        let gradient = shards(&outputs[1]).concat();
        assert_eq!(gradient.len(), inputs.len());
        for (gradient, input) in gradient.into_iter().zip(inputs) {
            assert!((gradient - f32::cos(input) * input * input).abs() <= 1e-5, "{gradient} for {input}");
        }
    }

    #[test]
    fn test_lowering_executes_shard_map_transpositions_for_callers_placed_along_manual_axes() {
        // The caller places its input along the manual axis `x` and the explicit axis `y`, while the map places it
        // along `x` only, so the transposed map assembles the cotangent along `x`. One placement-only `broadcast`
        // restores the caller's placement along both axes, which compiles to local slicing without any collective,
        // and every device receives its shard of the identity map's cotangent.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let domain = Session::new(&client).domain();
        let along_x = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let along_x_and_y = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let caller_type = ArrayType::new_static(DataType::F32, [8]).with_sharding(along_x_and_y).unwrap();
        let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
                let input =
                    ValueProjection::<ArrayType>::into_projected(inputs[0].clone()).map_err(ProgramError::from)?;
                let output = shard_map(
                    |x: ArrayTracer| Ok(x),
                    input,
                    logical_mesh.clone(),
                    along_x.clone(),
                    along_x.clone(),
                    ShardMapOptions::new(),
                )
                .unwrap();
                Ok(vec![output.into_value()])
            },
            vec![ArrayIrType::Array(caller_type.clone())],
        )
        .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap();
        let lowered = domain.lower_program(&pullback, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();

        // The serialized optimized HLO module names the opcode of every instruction (e.g., of its `parameter`), so it
        // names no collective opcode.
        let optimized = compiled.optimized_program().unwrap();
        let names_opcode =
            |opcode: &str| optimized.bytes.windows(opcode.len()).any(|window| window == opcode.as_bytes());
        assert!(names_opcode("parameter"));
        for opcode in ["all-gather", "all-reduce", "all-to-all", "collective-permute", "reduce-scatter"] {
            assert!(!names_opcode(opcode), "the optimized program contains a `{opcode}`");
        }

        let cotangent_type = ArrayType::new_static(DataType::F32, [8]).with_sharding(along_x).unwrap();
        let cotangent = Array::from_host_buffer(
            &domain,
            cotangent_type,
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).as_slice(),
        )
        .unwrap();
        let outputs = domain.execute_program(&compiled, vec![cotangent]).unwrap();
        assert_eq!(outputs[0].r#type().as_ref(), &caller_type);
        let shards = outputs[0]
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect::<Vec<_>>();
        assert_eq!(shards, vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]]);
    }

    #[test]
    fn test_lowering_executes_shard_maps_batched_along_every_manual_axis() {
        // The caller maps the leading dimension of its input, which it places along `x` over the mesh of the map
        // itself, over a map whose only active manual axis is `x`. The placement adds no manual variation to the batch
        // items, and batching releases `x`, which leaves a manual computation without manual axes that runs the batched
        // body once on the global values, while the boundary keeps the batch dimension placed along `x`, so each of the
        // two devices holds one batch item.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let domain = Session::new(&client).domain();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let input_sharding = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(input_sharding).unwrap();
        let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
                let output = batch(
                    |item| {
                        let item = ValueProjection::<ArrayType>::into_projected(item)?;
                        let output = shard_map(
                            |x| Ok(x.clone() + x),
                            item,
                            logical_mesh.clone(),
                            replicated.clone(),
                            replicated.clone(),
                            ShardMapOptions::new(),
                        )?;
                        Ok(output.into_value())
                    },
                    inputs[0].clone(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?;
                Ok(vec![output])
            },
            vec![ArrayIrType::Array(input_type.clone())],
        )
        .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(
            &*lowered.stable_hlo,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>}) -> \
                      (tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\", ?}, {}]>] \
                        out_shardings=[<@mesh, [{\"x\", ?}, {}]>] manual_axes={} (%arg1: tensor<2x3xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2x3xf32>
                      sdy.return %1 : tensor<2x3xf32>
                    } : (tensor<2x3xf32>) -> tensor<2x3xf32>
                    return %0 : tensor<2x3xf32>
                  }
                }
            "},
        );
        let compiled = domain.compile_program(&lowered).unwrap();
        let input = Array::from_host_buffer(
            &domain,
            input_type,
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).as_slice(),
        )
        .unwrap();
        let outputs = domain.execute_program(&compiled, vec![input]).unwrap();
        let items = outputs[0]
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect::<Vec<_>>();
        assert_eq!(items, vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]]);
    }

    #[test]
    fn test_lowering_accumulates_slice_cotangents_into_existing_state() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(replicated_vector_type(&mesh, 4).into());
        let output = builder
            .add_instruction(SliceOperation::new(vec![1], vec![3]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(
            pullback.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["reference_add_update"],
        );

        // Discharge reads only the selected buffer entries, adds the seed, and writes that slice back. The lowered
        // graph contains no full-size zero gradient or dense addition before the indexed update.
        let lowered = domain.lower_program(&pullback, 0, &Options::new(mesh)).unwrap();
        let operations = lowered
            .stable_hlo()
            .lines()
            .filter_map(|line| line.split_once(" = stablehlo."))
            .map(|(_, operation)| operation.split_whitespace().next().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(operations, vec!["slice", "add", "constant", "dynamic_update_slice"]);
    }

    #[test]
    fn test_lowering_discharges_local_index_and_slice_reference_transforms() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let vector_type = replicated_vector_type(&mesh, 4);
        let pair_type = replicated_vector_type(&mesh, 2);
        let mut builder = ProgramBuilder::new();
        let initial = builder.add_input(vector_type.into());
        let replacement = builder.add_input(pair_type.clone().into());
        let update = builder.add_input(pair_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let indexed_snapshot = builder
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(3),
                }]),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let transforms = vec![
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 3, 1)] },
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] },
        ];
        let old = builder
            .add_instruction(
                ReferenceSwapOperation::new().with_transforms(transforms.clone()),
                Vec::new(),
                vec![reference, replacement],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                ReferenceAddUpdateOperation::new().with_transforms(transforms),
                Vec::new(),
                vec![reference, update],
                None,
            )
            .unwrap();
        let final_snapshot =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![indexed_snapshot, old, final_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let expected_signature = concat!(
            "  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, ",
            "%arg1: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, ",
            "%arg2: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> ",
            "(tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, ",
            "tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {",
        );
        let expected_update = |result, target, update, index, target_size, update_size| {
            format!(
                "    %{result} = stablehlo.dynamic_update_slice %{target}, %{update}, %{index} : \
                 (tensor<{target_size}xf32>, tensor<{update_size}xf32>, tensor<i64>) -> \
                 tensor<{target_size}xf32>",
            )
        };
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                @SIGNATURE@
                    %0 = stablehlo.slice %arg0 [3:4] : (tensor<4xf32>) -> tensor<1xf32>
                    %1 = stablehlo.reshape %0 : (tensor<1xf32>) -> tensor<f32>
                    %2 = stablehlo.slice %arg0 [1:4] : (tensor<4xf32>) -> tensor<3xf32>
                    %3 = stablehlo.slice %2 [0:2] : (tensor<3xf32>) -> tensor<2xf32>
                    %c = stablehlo.constant dense<0> : tensor<i64>
                @UPDATE0@
                    %c_0 = stablehlo.constant dense<1> : tensor<i64>
                @UPDATE1@
                    %6 = stablehlo.slice %5 [1:4] : (tensor<4xf32>) -> tensor<3xf32>
                    %7 = stablehlo.slice %6 [0:2] : (tensor<3xf32>) -> tensor<2xf32>
                    %8 = stablehlo.add %7, %arg2 : tensor<2xf32>
                    %c_1 = stablehlo.constant dense<0> : tensor<i64>
                @UPDATE2@
                    %c_2 = stablehlo.constant dense<1> : tensor<i64>
                @UPDATE3@
                    return %1, %3, %10 : tensor<f32>, tensor<2xf32>, tensor<4xf32>
                  }
                }
            "#}
            .replace("@SIGNATURE@", expected_signature)
            .replace("@UPDATE0@", &expected_update("4", "2", "arg1", "c", 3, 2))
            .replace("@UPDATE1@", &expected_update("5", "arg0", "4", "c_0", 4, 3))
            .replace("@UPDATE2@", &expected_update("9", "6", "8", "c_1", 3, 2))
            .replace("@UPDATE3@", &expected_update("10", "5", "9", "c_2", 4, 3)),
        );
        let compiled = domain.compile_program(&lowered).unwrap();
        let outputs = domain
            .execute_program(
                &compiled,
                vec![
                    f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0, 4.0]),
                    f32_vector(&domain, &mesh, &[10.0, 20.0]),
                    f32_vector(&domain, &mesh, &[1.0, 2.0]),
                ],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, &outputs[0]), vec![4.0]);
        assert_eq!(read_f32s(&client, &outputs[1]), vec![2.0, 3.0]);
        assert_eq!(read_f32s(&client, &outputs[2]), vec![1.0, 11.0, 22.0, 4.0]);
    }

    #[test]
    fn test_lowering_discharges_and_executes_a_write_through_a_reference_view() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let vector_type = replicated_vector_type(&mesh, 4);
        let pair_type = replicated_vector_type(&mesh, 2);
        let mut builder = ProgramBuilder::new();
        let initial = builder.add_input(vector_type.into());
        let replacement = builder.add_input(pair_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(
                ReferenceWriteOperation::new()
                    .with_transforms(vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] }]),
                Vec::new(),
                vec![reference, replacement],
                None,
            )
            .unwrap();
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program: FlatProgram = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let outputs = domain
            .execute_program(
                &compiled,
                vec![f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0, 4.0]), f32_vector(&domain, &mesh, &[10.0, 20.0])],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, &outputs[0]), vec![1.0, 10.0, 20.0, 4.0]);
    }

    #[test]
    fn test_lowering_executes_indexed_reference_mutation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let vector_type = replicated_vector_type(&mesh, 3);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let mut builder = ProgramBuilder::new();
        let initial = builder.add_input(vector_type.into());
        let replacement = builder.add_input(scalar_type.clone().into());
        let update = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) }];
        let old = builder
            .add_instruction(
                ReferenceSwapOperation::new().with_transforms(transforms.clone()),
                Vec::new(),
                vec![reference, replacement],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                ReferenceAddUpdateOperation::new().with_transforms(transforms),
                Vec::new(),
                vec![reference, update],
                None,
            )
            .unwrap();
        let final_snapshot =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![old, final_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let expected_signature = concat!(
            "  func.func @main(%arg0: tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, ",
            "%arg1: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "%arg2: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> ",
            "(tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {",
        );
        let expected_update = |result, target, update, index| {
            format!(
                "    %{result} = stablehlo.dynamic_update_slice %{target}, %{update}, %{index} : \
                 (tensor<3xf32>, tensor<1xf32>, tensor<i64>) -> tensor<3xf32>",
            )
        };
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                @SIGNATURE@
                    %0 = stablehlo.slice %arg0 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
                    %1 = stablehlo.reshape %0 : (tensor<1xf32>) -> tensor<f32>
                    %2 = stablehlo.reshape %arg1 : (tensor<f32>) -> tensor<1xf32>
                    %c = stablehlo.constant dense<1> : tensor<i64>
                @UPDATE0@
                    %4 = stablehlo.slice %3 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
                    %5 = stablehlo.reshape %4 : (tensor<1xf32>) -> tensor<f32>
                    %6 = stablehlo.add %5, %arg2 : tensor<f32>
                    %7 = stablehlo.reshape %6 : (tensor<f32>) -> tensor<1xf32>
                    %c_0 = stablehlo.constant dense<1> : tensor<i64>
                @UPDATE1@
                    return %1, %8 : tensor<f32>, tensor<3xf32>
                  }
                }
            "#}
            .replace("@SIGNATURE@", expected_signature)
            .replace("@UPDATE0@", &expected_update("3", "arg0", "2", "c"))
            .replace("@UPDATE1@", &expected_update("8", "3", "7", "c_0")),
        );
        let compiled = domain.compile_program(&lowered).unwrap();
        let outputs = domain
            .execute_program(
                &compiled,
                vec![
                    f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0]),
                    f32_scalar(&domain, &mesh, 10.0),
                    f32_scalar(&domain, &mesh, 3.0),
                ],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, &outputs[0]), vec![2.0]);
        assert_eq!(read_f32s(&client, &outputs[1]), vec![1.0, 13.0, 3.0]);
    }

    #[test]
    fn test_lowering_preserves_lifted_capture_scope_inside_condition_regions() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let predicate_type = replicated_scalar_type(&mesh, DataType::Boolean);

        let branch = || {
            let mut builder = ProgramBuilder::new();
            let capture =
                builder.add_constant(Constant::Captured(CaptureReference::new(0, scalar_type.clone().into())));
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![capture], Vec::new(), vec![Placeholder]).unwrap()
        };
        let true_branch = branch();
        let false_branch = branch();
        let mut builder = ProgramBuilder::new();
        builder.add_input(scalar_type.clone().into());
        let predicate = builder.add_input(predicate_type.into());
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![true_region, false_region], vec![predicate], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let lowered = domain.lower_program(&program, 1, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let outputs = domain
            .execute_program(&compiled, vec![f32_scalar(&domain, &mesh, 7.0), boolean_scalar(&domain, &mesh, true)])
            .unwrap();
        assert_eq!(read_f32s(&client, &outputs[0]), vec![7.0]);
    }

    #[test]
    fn test_lowering_validates_capture_count_before_reference_discharge() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(scalar_type.into());
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            domain.lower_program(&program, 2, &Options::new(mesh)),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::InvalidArgument { message })
                    if *message == "capture_count is 2, but the program has only 1 flat input(s)",
            ),
        ));
    }

    #[test]
    fn test_lowering_discharges_local_reference_state_through_condition() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let true_branch = {
            let mut builder = ProgramBuilder::new();
            let reference = builder.add_input(reference_type.clone().into());
            let replacement = builder.add_input(scalar_type.clone().into());
            let snapshot = builder
                .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::new();
            let reference = builder.add_input(reference_type.into());
            builder.add_input(scalar_type.clone().into());
            let snapshot =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(replicated_scalar_type(&mesh, DataType::Boolean).into());
        let initial = builder.add_input(scalar_type.clone().into());
        let replacement = builder.add_input(scalar_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, replacement],
                None,
            )
            .unwrap()[0];
        let final_value =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![snapshot, final_value],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.output_types(), &[scalar_type.clone(), scalar_type]);
        let expected_signature = concat!(
            "  func.func @main(%arg0: tensor<i1> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "%arg1: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "%arg2: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> ",
            "(tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) {",
        );
        let expected_stable_hlo = indoc! {r#"
            module {
              sdy.mesh @mesh = <["x"=1]>
            @SIGNATURE@
                %0:2 = "stablehlo.if"(%arg0) ({
                  stablehlo.return %arg1, %arg2 : tensor<f32>, tensor<f32>
                }, {
                  stablehlo.return %arg1, %arg1 : tensor<f32>, tensor<f32>
                }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>)
                return %0#0, %0#1 : tensor<f32>, tensor<f32>
              }
            }
        "#}
        .replace("@SIGNATURE@", expected_signature);
        assert_eq!(lowered.stable_hlo(), expected_stable_hlo,);
        let compiled = domain.compile_program(&lowered).unwrap();

        let true_outputs = domain
            .execute_program(
                &compiled,
                vec![
                    boolean_scalar(&domain, &mesh, true),
                    f32_scalar(&domain, &mesh, 10.0),
                    f32_scalar(&domain, &mesh, 7.0),
                ],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, &true_outputs[0]), vec![10.0]);
        assert_eq!(read_f32s(&client, &true_outputs[1]), vec![7.0]);

        let false_outputs = domain
            .execute_program(
                &compiled,
                vec![
                    boolean_scalar(&domain, &mesh, false),
                    f32_scalar(&domain, &mesh, 10.0),
                    f32_scalar(&domain, &mesh, 7.0),
                ],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, &false_outputs[0]), vec![10.0]);
        assert_eq!(read_f32s(&client, &false_outputs[1]), vec![10.0]);
    }

    #[test]
    fn test_lowering_discharges_local_reference_state_through_while() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let condition = {
            let mut builder = ProgramBuilder::new();
            let reference = builder.add_input(reference_type.clone().into());
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap();
            let predicate =
                CoreArray::from_elements(replicated_scalar_type(&mesh, DataType::Boolean), &[true]).unwrap();
            let predicate =
                builder.add_instruction(ConstantOperation::new(predicate), Vec::new(), Vec::new(), None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::new();
            let reference = builder.add_input(reference_type.into());
            let update = builder
                .add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(3).unwrap();
        let reference = builder.add_instruction(operation, vec![condition, body], vec![reference], None).unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.while").count(), 1, "{}", lowered.stable_hlo());
        let compiled = domain.compile_program(&lowered).unwrap();
        let outputs = domain.execute_program(&compiled, vec![f32_scalar(&domain, &mesh, 2.0)]).unwrap();

        // Three bounded iterations accumulate the constant `1.0` into the initial `2.0` state.
        assert_eq!(read_f32s(&client, &outputs[0]), vec![5.0]);
    }

    #[test]
    fn test_lowering_discharges_local_reference_state_through_static_scan() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let body = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let reference = builder.add_input(reference_type.into());
            let update = builder
                .add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let value =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![reference, value],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };

        let mut builder = ProgramBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let scan_outputs = builder
            .add_instruction(ScanOperation::new(1, 3), vec![body], vec![reference], None)
            .unwrap()
            .to_vec();
        let final_value = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![scan_outputs[0]], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![final_value, scan_outputs[1]],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.while").count(), 1, "{}", lowered.stable_hlo());
        let compiled = domain.compile_program(&lowered).unwrap();
        let outputs = domain.execute_program(&compiled, vec![f32_scalar(&domain, &mesh, 2.0)]).unwrap();

        // Three scan iterations accumulate the constant `1.0` into the initial `2.0` state, and the body's second
        // output stacks the state observed after each iteration.
        assert_eq!(read_f32s(&client, &outputs[0]), vec![5.0]);
        assert_eq!(read_f32s(&client, &outputs[1]), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_lowering_discharges_local_reference_state_through_dynamic_scan() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let length_variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        let length = DimensionValue::new(DimensionType::from(length_variable.clone()), 3).unwrap();

        let body = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let reference = builder.add_input(reference_type.into());
            let update = builder
                .add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![reference],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };

        let mut builder = ProgramBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let runtime_length = builder.add_constant(Constant::Dimension(length.clone()));
        let reference = builder
            .add_instruction(
                ScanOperation::new(1, CoreDimension::Dynamic(length_variable.clone())),
                vec![body],
                vec![reference, runtime_length],
                None,
            )
            .unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let scan_instruction = &program.entry_region_ref().instructions()[1];
        let Operation::Scan(scan) = scan_instruction.operation() else {
            panic!("expected dynamic scan operation");
        };
        assert_eq!(scan.carry_count(), 1);
        assert_eq!(scan.iteration_count(), &CoreDimension::Dynamic(length_variable));
        let runtime_length = *scan_instruction.inputs().last().unwrap();
        assert_eq!(
            program.entry_region_ref().atoms()[runtime_length.index()].as_constant(),
            Some(&Constant::Dimension(length)),
        );
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.while").count(), 1, "{}", lowered.stable_hlo());
        let compiled = domain.compile_program(&lowered).unwrap();
        let outputs = domain.execute_program(&compiled, vec![f32_scalar(&domain, &mesh, 2.0)]).unwrap();

        // The runtime length is three, so the scan accumulates the constant `1.0` into the initial `2.0` state three
        // times.
        assert_eq!(read_f32s(&client, &outputs[0]), vec![5.0]);
    }

    #[test]
    fn test_lowering_discharges_local_reference_state_through_nested_jit_call() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let callee = {
            let mut builder = ProgramBuilder::new();
            let reference = builder.add_input(reference_type.into());
            let update = builder.add_input(scalar_type.clone().into());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![reference], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let initial = builder.add_input(scalar_type.clone().into());
        let update = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let reference = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![reference, update], None)
            .unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            program.entry_region_ref().instructions()[1].operation(),
            Operation::JitCall(operation) if operation.capture_count() == 0,
        ));
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.add").count(), 1, "{}", lowered.stable_hlo());
        let compiled = domain.compile_program(&lowered).unwrap();
        let outputs = domain
            .execute_program(&compiled, vec![f32_scalar(&domain, &mesh, 2.0), f32_scalar(&domain, &mesh, 4.0)])
            .unwrap();

        let immutable_oracle = 2.0f32 + 4.0;
        assert_eq!(read_f32s(&client, &outputs[0]), vec![immutable_oracle]);
    }

    #[test]
    fn test_lowering_carries_external_reference_state_metadata_after_discharge() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();

        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let output = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program: FlatProgram = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh)).unwrap();
        assert_eq!(lowered.output_count, 1);
        assert_eq!(lowered.reference_states.len(), 1);
        assert_eq!(lowered.reference_states[0].source().flat_input_index(lowered.capture_count).unwrap(), 0);
        assert_eq!(lowered.reference_states[0].output_index(), None);
        assert!(!lowered.stable_hlo.contains("reference"));
    }

    #[test]
    fn test_lowering_preserves_provenance_locations_end_to_end() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::F64);

        // Builds `-(x * x)` with the multiplication carrying the provided provenance.
        fn build_program(
            input_type: &ArrayType,
            provenance: Provenance,
        ) -> crate::experimental::ops::Program<Vec<Constant>, Vec<Constant>> {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(input_type.clone().into());
            let product = builder
                .add_instruction(MulOperation::new(), Vec::new(), vec![input, input], Some(provenance))
                .unwrap()[0];
            let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![product], None).unwrap()[0];
            builder.build(vec![negated], vec![Placeholder], vec![Placeholder]).unwrap()
        }

        let coordinate_basis = Provenance::scope(
            ProvenanceScope::new("ryft"),
            Provenance::scope(
                ProvenanceScope::new("differentiation"),
                Provenance::scope(ProvenanceScope::new("coordinate_basis"), Provenance::unknown()),
            ),
        );
        let scoped = build_program(&input_type, coordinate_basis);
        let options = Options::new(mesh.clone());
        let lowered = domain.lower_program(&scoped, 0, &options).unwrap();

        // The nested framework scopes survive lowering as a chain of nested name locations that is visible in the
        // serialized module (and therefore in dumped MLIR/HLO metadata), while the unknown-provenance negation and
        // the module scaffolding keep the base location.
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                #loc = loc(unknown)
                module {
                  sdy.mesh @mesh = <["x"=1]> loc(#loc)
                  func.func @main(%arg0: tensor<f64> {sdy.sharding = #sdy.sharding<@mesh, []>} loc(unknown)) -> (tensor<f64> {sdy.sharding = #sdy.sharding<@mesh, []>}) {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64> loc(#loc3)
                    %1 = stablehlo.negate %0 : tensor<f64> loc(#loc)
                    return %1 : tensor<f64> loc(#loc)
                  } loc(#loc)
                } loc(#loc)
                #loc1 = loc("coordinate_basis")
                #loc2 = loc("differentiation"(#loc1))
                #loc3 = loc("ryft"(#loc2))
            "#},
        );

        // Provenance does not affect numeric results.
        let compiled = domain.compile_program(&lowered).unwrap();
        let value = Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), 3.0f64.to_ne_bytes().as_slice())
            .unwrap();
        let outputs = domain.execute_program(&compiled, vec![value]).unwrap();
        assert_eq!(read_f64s(&client, &outputs[0]), vec![-9.0]);

        // Two semantically equal programs with different scope names serialize differently and therefore key
        // differently, so neither can receive the other's labeled executable, while a provenance-free program keeps
        // the location-free serialization (and hence its pre-provenance cache key) entirely.
        let relabeled =
            build_program(&input_type, Provenance::scope(ProvenanceScope::new("other_label"), Provenance::unknown()));
        let relabeled = domain.lower_program(&relabeled, 0, &options).unwrap();
        assert_ne!(lowered.compilation_key().unwrap(), relabeled.compilation_key().unwrap(),);
        let unlabeled = build_program(&input_type, Provenance::unknown());
        let unlabeled = domain.lower_program(&unlabeled, 0, &options).unwrap();
        assert_eq!(
            unlabeled.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<f64> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> (tensor<f64> {sdy.sharding = #sdy.sharding<@mesh, []>}) {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
                    %1 = stablehlo.negate %0 : tensor<f64>
                    return %1 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_canonical_compilation_options_sorts_protobuf_maps() {
        use ryft_pjrt::protos::{DebugOptions, ExecutableCompilationOptions, OptionOverride};

        let mut left = CompilationOptions::default();
        left.environment_option_overrides.insert("z".into(), OptionOverride::default());
        left.environment_option_overrides.insert("a".into(), OptionOverride::default());
        left.executable_build_options = Some(ExecutableCompilationOptions {
            debug_options: Some(DebugOptions {
                xla_backend_extra_options: [("z".into(), "2".into()), ("a".into(), "1".into())].into_iter().collect(),
                ..DebugOptions::default()
            }),
            ..ExecutableCompilationOptions::default()
        });
        let mut right = CompilationOptions::default();
        right.environment_option_overrides.insert("a".into(), OptionOverride::default());
        right.environment_option_overrides.insert("z".into(), OptionOverride::default());
        right.executable_build_options = Some(ExecutableCompilationOptions {
            debug_options: Some(DebugOptions {
                xla_backend_extra_options: [("a".into(), "1".into()), ("z".into(), "2".into())].into_iter().collect(),
                ..DebugOptions::default()
            }),
            ..ExecutableCompilationOptions::default()
        });

        assert_eq!(left, right);
        assert_eq!(canonical_compilation_options_bytes(&left), canonical_compilation_options_bytes(&right));
    }

    #[test]
    fn test_ragged_dot_lowering_strategy_is_compilation_policy() {
        use std::collections::HashMap;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let lhs_type = ArrayType::new_static(DataType::F32, [5, 2]);
        let rhs_type = ArrayType::new_static(DataType::F32, [3, 2, 1]);
        let group_sizes_type = ArrayType::new_static(DataType::I32, [3]);
        let mut builder = ProgramBuilder::new();
        let lhs = builder.add_input(ArrayIrType::Array(lhs_type));
        let rhs = builder.add_input(ArrayIrType::Array(rhs_type));
        let group_sizes = builder.add_input(ArrayIrType::Array(group_sizes_type));
        let output = builder
            .add_instruction(
                RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
                Vec::new(),
                vec![lhs, rhs, group_sizes],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let decomposition_options = Options::new(mesh.clone());
        let instruction_options =
            Options::new(mesh).with_ragged_dot_lowering_strategy(RaggedDotLoweringStrategy::Instruction);
        let decomposition = domain.lower_program(&program, 0, &decomposition_options).unwrap();
        let instruction = domain.lower_program(&program, 0, &instruction_options).unwrap();

        assert!(decomposition.stable_hlo().contains("stablehlo.dot_general"));
        assert!(!decomposition.stable_hlo().contains("chlo.ragged_dot"));
        assert!(instruction.stable_hlo().contains("chlo.ragged_dot"));
        assert_ne!(decomposition.stable_hlo(), instruction.stable_hlo());
        assert_ne!(decomposition_options, instruction_options);
        let options_by_representation = HashMap::from([
            (decomposition_options.clone(), "decomposition"),
            (instruction_options.clone(), "instruction"),
        ]);
        assert_eq!(options_by_representation.len(), 2);
        assert_eq!(options_by_representation.get(&decomposition_options), Some(&"decomposition"));
        assert_eq!(options_by_representation.get(&instruction_options), Some(&"instruction"));
    }

    #[test]
    fn test_data_dependent_padding_inventory_is_kind_and_kernel_specific() {
        use DataDependentPaddingDiscipline::{Propagated, RyftMasked, Unsupported, XlaMasked};

        // Every reference operation shares one collapsed match arm, so one representative pins that arm.
        assert_eq!(
            data_dependent_padding_discipline(&Operation::ReferenceNew(ReferenceNewOperation::new())),
            Unsupported { reason: "references must be discharged before bounded-dynamic XLA validation" },
        );

        for kind in [
            ReductionKind::Sum,
            ReductionKind::Product,
            ReductionKind::LogSumExp,
            ReductionKind::Max,
            ReductionKind::Min,
            ReductionKind::Any,
            ReductionKind::All,
        ] {
            assert_eq!(reduction_data_dependent_padding_discipline(kind), XlaMasked);
        }
        assert_eq!(
            reduction_data_dependent_padding_discipline(ReductionKind::Mean),
            Unsupported { reason: "mean over a dynamically sized reduction axis is not supported by the XLA lowering" },
        );
        for direction in [SortDirection::Ascending, SortDirection::Descending] {
            assert_eq!(sort_data_dependent_padding_discipline(direction), XlaMasked);
        }
        for operation in [
            ArrayOperation::ArgMax(ArgMaxOperation::new(0, DataType::I32)),
            ArrayOperation::ArgMin(ArgMinOperation::new(0, DataType::I32)),
        ] {
            assert_eq!(array_data_dependent_padding_discipline(&operation), XlaMasked);
        }
        for kind in [
            ScatterReductionKind::Overwrite,
            ScatterReductionKind::Add,
            ScatterReductionKind::Mul,
            ScatterReductionKind::Min,
            ScatterReductionKind::Max,
        ] {
            assert!(matches!(
                scatter_data_dependent_padding_discipline(kind),
                Unsupported { reason } if reason.contains("scatter padding semantics have not been verified"),
            ));
        }

        assert_eq!(
            array_data_dependent_padding_discipline(&ArrayOperation::ScaledDot(ScaledDotOperation::new(
                DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                DataType::F32,
                true,
                true,
            ))),
            RyftMasked,
        );

        // A log-sum-exp `reduce` expands into maximum and summation reductions, so it carries the same masking
        // contract as those two reduction kinds.
        assert_eq!(
            array_data_dependent_padding_discipline(&ArrayOperation::Reduce(ReduceOperation::new(
                vec![0],
                ReductionKind::LogSumExp,
            ))),
            XlaMasked,
        );

        // The prefix scans own their own arm rather than sharing the collapsed shape-preserving one, because the
        // classification rests on type inference keeping the scanned axis static.
        for kind in [
            CumulativeKind::Sum,
            CumulativeKind::Product,
            CumulativeKind::Max,
            CumulativeKind::Min,
            CumulativeKind::LogSumExp,
        ] {
            assert_eq!(
                array_data_dependent_padding_discipline(&ArrayOperation::Cumulative(CumulativeOperation::new(0, kind))),
                Propagated,
            );
        }
    }

    #[test]
    fn test_condition_recreates_reference_view_inside_branch() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let vector_type = replicated_vector_type(&mesh, 3);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(vector_type.clone());
        let true_branch = {
            let mut builder = ProgramBuilder::new();
            let reference = builder.add_input(reference_type.clone().into());
            let update =
                builder.add_instruction(OneOperation::new(scalar_type), Vec::new(), Vec::new(), None).unwrap()[0];
            builder
                .add_instruction(
                    ReferenceAddUpdateOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                        axis: 0,
                        index: ArrayReferenceTransformIndex::Static(1),
                    }]),
                    Vec::new(),
                    vec![reference, update],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::new();
            let reference = builder.add_input(reference_type.into());
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(replicated_scalar_type(&mesh, DataType::Boolean).into());
        let initial = builder.add_input(vector_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let reference = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let expected_signature = concat!(
            "  func.func @main(%arg0: tensor<i1> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "%arg1: tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> ",
            "(tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {",
        );
        let expected_update = concat!(
            "      %5 = stablehlo.dynamic_update_slice %arg1, %4, %c : ",
            "(tensor<3xf32>, tensor<1xf32>, tensor<i64>) -> tensor<3xf32>",
        );
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                @SIGNATURE@
                    %0 = "stablehlo.if"(%arg0) ({
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                      %1 = stablehlo.slice %arg1 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
                      %2 = stablehlo.reshape %1 : (tensor<1xf32>) -> tensor<f32>
                      %3 = stablehlo.add %2, %cst : tensor<f32>
                      %4 = stablehlo.reshape %3 : (tensor<f32>) -> tensor<1xf32>
                      %c = stablehlo.constant dense<1> : tensor<i64>
                @UPDATE@
                      stablehlo.return %5 : tensor<3xf32>
                    }, {
                      stablehlo.return %arg1 : tensor<3xf32>
                    }) : (tensor<i1>) -> tensor<3xf32>
                    return %0 : tensor<3xf32>
                  }
                }
            "#}
            .replace("@SIGNATURE@", expected_signature)
            .replace("@UPDATE@", expected_update),
        );
        let compiled = domain.compile_program(&lowered).unwrap();
        for (predicate, expected) in [(true, vec![1.0, 3.0, 3.0]), (false, vec![1.0, 2.0, 3.0])] {
            let outputs = domain
                .execute_program(
                    &compiled,
                    vec![boolean_scalar(&domain, &mesh, predicate), f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0])],
                )
                .unwrap();
            assert_eq!(read_f32s(&client, &outputs[0]), expected);
        }
    }

    #[test]
    fn test_compiled_dynamic_slice_runtime_size_pullback() {
        // The size comes from device data and changes without recompilation. Seed the pullback with the sliced
        // values so this fixture tests both the strided forward and point-scatter adjoint, including an empty result.
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = replicated_vector_type(&mesh, 6);
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let size = builder.add_input(size_type.clone().into());
        let size = builder
            .add_instruction(
                DimensionFromScalarOperation::new(
                    DimensionVariable::new("size", DimensionBounds::new(0, Some(4)).unwrap()).into(),
                ),
                Vec::new(),
                vec![size],
                None,
            )
            .unwrap()[0];
        let start = builder.add_constant(Constant::Dimension(DimensionValue::constant(9).unwrap()));
        let output = builder
            .add_instruction(
                DynamicSliceOperation::<ArrayIrType>::from_rank(1)
                    .with_strides(vec![2])
                    .unwrap()
                    .with_bounds(ryft_core::DynamicSliceBounds::Clamp),
                Vec::new(),
                vec![input, start, size],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        let pullback = linearization.pullback().unwrap().into_program();
        let context = TracingContext::<Constant, Operation>::new();
        let inputs = vec![context.input(input_type.clone().into()), context.input(size_type.clone().into())];
        let mut outputs = linearization.primal().interpret_in_context(&context, inputs).unwrap();
        let mut arguments = vec![outputs[0].clone()];
        arguments.extend(outputs.split_off(1));
        outputs.extend(pullback.interpret_in_context(&context, arguments).unwrap());
        let combined = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Constant>, Vec<Constant>>(
                outputs.iter().map(|output| output.atom_id().unwrap()).collect(),
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let lowered = domain.lower_program(&combined, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let input =
            Array::from_host_buffer(&domain, input_type, mesh.clone(), values_to_bytes(&[0f32, 1., 2., 3., 4., 5.]))
                .unwrap();
        let size = Array::from_host_buffer(&domain, size_type.clone(), mesh.clone(), values_to_bytes(&[2i64])).unwrap();
        let outputs = domain.execute_program(&compiled, vec![input.clone(), size]).unwrap();
        assert_eq!(outputs[0].shape().as_slice(), &[2]);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![3., 5.]);
        assert_eq!(read_f32s(&client, &outputs[1]), vec![0., 0., 0., 3., 0., 5.]);
        let size = Array::from_host_buffer(&domain, size_type, mesh, values_to_bytes(&[0i64])).unwrap();
        let outputs = domain.execute_program(&compiled, vec![input, size]).unwrap();
        assert_eq!(outputs[0].shape().as_slice(), &[0]);
        assert_eq!(read_f32s(&client, &outputs[1]), vec![0f32; 6]);
    }

    #[test]
    fn test_compiled_dynamic_slice_runtime_input_pullback() {
        use ryft_core::{Add, DimensionConstant, Div};

        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = ArrayType::new_static(DataType::F32, [4, 5]);
        let scalar_type = ArrayType::scalar(DataType::I64);
        let context = TracingContext::<Constant, Operation>::new();
        let input = context.input(input_type.clone().into());
        let rows = context
            .input(scalar_type.clone().into())
            .to_dimension(DimensionVariable::new("rows", DimensionBounds::new(0, Some(5)).unwrap()))
            .unwrap();
        let columns = context
            .input(scalar_type.clone().into())
            .to_dimension(DimensionVariable::new("columns", DimensionBounds::new(0, Some(6)).unwrap()))
            .unwrap();
        let zero = context.dimension_constant(0).unwrap();
        let one = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(1).unwrap()).unwrap();
        let two = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(2).unwrap()).unwrap();
        let height = ValueProjection::<DimensionType>::into_projected(rows.clone()).unwrap();
        let height = height.add(&one).unwrap().div(&two).unwrap().into_value();
        let width = ValueProjection::<DimensionType>::into_projected(columns.clone()).unwrap();
        let width = width.add(&one).unwrap().div(&two).unwrap().into_value();
        // The inner slice receives a genuinely runtime-shaped array. Its pullback must allocate that exact shape,
        // before the outer prefix slice embeds it back into the original static input shape.
        let prefix = input
            .dynamic_slice_with_dimensions(&[zero.clone(), zero.clone()], &[rows, columns], &[1, 1])
            .unwrap();
        let output = prefix.dynamic_slice_with_dimensions(&[zero.clone(), zero], &[height, width], &[2, 2]).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        let pullback = linearization.pullback().unwrap().into_program();
        let context = TracingContext::<Constant, Operation>::new();
        let inputs = vec![
            context.input(input_type.clone().into()),
            context.input(scalar_type.clone().into()),
            context.input(scalar_type.clone().into()),
        ];
        let mut outputs = linearization.primal().interpret_in_context(&context, inputs).unwrap();
        let mut arguments = vec![outputs[0].clone()];
        arguments.extend(outputs.split_off(1));
        // Replaying arithmetic dimensions creates fresh identities. Instantiate this standalone pullback's
        // formal dimension identities for the replayed residual signature before importing its operations.
        let input_types = arguments.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let pullback = pullback.with_instantiated_type_identities(&input_types).unwrap();
        outputs.extend(pullback.interpret_in_context(&context, arguments).unwrap());
        let combined = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Constant>, Vec<Constant>>(
                outputs.iter().map(|output| output.atom_id().unwrap()).collect(),
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let lowered = domain.lower_program(&combined, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let values = [
            0f32,
            1.,
            2.,
            3.,
            f32::NAN,
            5.,
            6.,
            7.,
            8.,
            f32::NAN,
            10.,
            11.,
            12.,
            13.,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        ];
        let input = Array::from_host_buffer(&domain, input_type, mesh.clone(), values_to_bytes(&values)).unwrap();
        let rows =
            Array::from_host_buffer(&domain, scalar_type.clone(), mesh.clone(), values_to_bytes(&[3i64])).unwrap();
        let columns =
            Array::from_host_buffer(&domain, scalar_type.clone(), mesh.clone(), values_to_bytes(&[4i64])).unwrap();
        let outputs = domain.execute_program(&compiled, vec![input.clone(), rows, columns.clone()]).unwrap();
        assert_eq!(outputs[0].shape().as_slice(), &[2, 2]);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![0., 2., 10., 12.]);
        assert_eq!(
            read_f32s(&client, &outputs[1]),
            vec![0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 10., 0., 12., 0., 0., 0., 0., 0., 0., 0.]
        );
        let rows = Array::from_host_buffer(&domain, scalar_type, mesh, values_to_bytes(&[0i64])).unwrap();
        let outputs = domain.execute_program(&compiled, vec![input, rows, columns]).unwrap();
        assert_eq!(outputs[0].shape().as_slice(), &[0, 2]);
        assert_eq!(read_f32s(&client, &outputs[1]), vec![0f32; 20]);
    }

    #[test]
    fn test_shard_map_emulation_matches_execution_on_cpu() {
        // The reference backend's lockstep emulation of `shard_map` and XLA's manual computation agree exactly on
        // boundary splitting and assembly, on every collective over the manual axis, and on reference accesses and
        // write-back (including reads of a replicated reference at device-varying indices and invariant mutations of a
        // replicated reference).
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::IdentityAndSum);
        assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::AxisIndex);
        assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::AllGather);
        assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::AllToAll);
        assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::Permute);
        assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::Reference);
        assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::VaryingIndexReference);
        assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::ReplicatedReference);
    }

    #[test]
    fn test_shard_map_pullback_emulation_matches_execution_on_cpu() {
        // The pullback of a read of a replicated reference at device-varying indices scatters each device's cotangent
        // into a local buffer that varies along `x`, sums the buffers across `x`, and adds the sum into the replicated
        // cotangent reference, both in the emulation and in XLA's manual computation. The pullbacks of an update of a
        // sharded reference and of an invariant mutation of a replicated reference agree as well.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        assert_shard_map_pullback_emulation_matches_execution(&client, ShardMapDifferentialCase::Reference);
        let case = ShardMapDifferentialCase::VaryingIndexReference;
        assert_shard_map_pullback_emulation_matches_execution(&client, case);
        let case = ShardMapDifferentialCase::ReplicatedReference;
        assert_shard_map_pullback_emulation_matches_execution(&client, case);
    }

    #[test]
    fn test_nested_shard_map_emulation_matches_execution_on_cpu() {
        // A nested map whose inner body sums across the enclosing manual axis runs on all four devices of a 2×2 mesh,
        // both in the emulation and in XLA's nested manual computations.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::Nested);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_cuda_13_plugin_executes_bounded_dynamic_program_at_two_sizes() {
        fn trace_data_derived<'c>(
            extent: DimensionVariable,
            inputs: Vec<CompilationTracer<Domain<'c>>>,
        ) -> Result<CompilationTracer<Domain<'c>>, Error> {
            let [size, value] = inputs.as_slice() else {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
            };
            let dimension = size.to_dimension(extent)?;
            let broadcast = size
                .context()
                .bind(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), &[value.clone(), dimension])?
                .remove(0);
            Ok(size
                .context()
                .bind(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), &[broadcast])?
                .remove(0))
        }

        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |input| input.clone() + input,
            dynamic_type,
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let compiled: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
            staged.lower(&domain).unwrap().compile(&domain).unwrap();
        assert_eq!(domain.program_count(), 1);

        for size in [4usize, 7] {
            let input_type = replicated_vector_type(&mesh, size);
            let values = (0..size).map(|value| value as f32).collect::<Vec<_>>();
            let input = Array::from_host_buffer(
                &domain,
                input_type,
                mesh.clone(),
                values_to_bytes(values.as_slice()).as_slice(),
            )
            .unwrap();
            for _ in 0..2 {
                // Both launches use the same bounded executable. The second also proves that replay does not
                // accidentally specialize or recompile the program for the concrete logical size.
                let output = compiled.executable().execute(&domain, Value::Array(input.clone())).unwrap();
                let Value::Array(output) = output else {
                    panic!("array-only compiled function returned a first-class dimension");
                };
                output.block_until_ready().unwrap();
                assert_eq!(output.shape().as_slice(), &[size]);
                assert_eq!(read_f32s(&client, &output), values.iter().map(|value| value * 2.0).collect::<Vec<_>>());
                assert_eq!(domain.program_count(), 1, "executing a loaded executable must not trigger recompilation");
            }
        }

        // The same plugin route also accepts an extent born from device data inside the program. The gateway's
        // ordered bounds assertion and the reduction's dynamic masking both survive compilation, and the scalar input
        // values select different logical extents without producing another retained specialization.
        let scalar_i64 = replicated_scalar_type(&mesh, DataType::I64);
        let scalar_f32 = replicated_scalar_type(&mesh, DataType::F32);
        let extent = DimensionVariable::new("data_extent", DimensionBounds::new(1, Some(8)).unwrap());
        let function: CompiledFunctionDispatcher<Domain<'_>, (), Vec<ArrayIrType>, ArrayIrType, _> = domain
            .jit_with_options(move |(), inputs| trace_data_derived(extent.clone(), inputs), Options::new(mesh.clone()));
        let call = |size: i64| {
            let size =
                Array::from_host_buffer(&domain, scalar_i64.clone(), mesh.clone(), size.to_ne_bytes().as_slice())
                    .unwrap();
            let value =
                Array::from_host_buffer(&domain, scalar_f32.clone(), mesh.clone(), 2.0f32.to_ne_bytes().as_slice())
                    .unwrap();
            function.execute((), vec![Value::Array(size), Value::Array(value)])
        };
        assert_eq!(read_f32s(&client, program_array(&call(4).unwrap())), vec![8.0]);
        assert_eq!(read_f32s(&client, program_array(&call(7).unwrap())), vec![14.0]);
        let invalid = call(8).unwrap();
        let error = program_array(&invalid).block_until_ready().unwrap_err();
        assert!(
            error.to_string().contains(
                "`dimension_from_scalar` failed: input dimension `data_extent` = 8 is outside its declared bounds \
                 [1, 8)",
            ),
            "{error}",
        );
        assert_eq!(function.executable_specialization_count(), 1);
        assert_eq!(function.statistics().dispatch_misses, 1);
        assert_eq!(function.statistics().dispatch_hits, 2);
    }

    #[test]
    fn test_data_derived_extent_retained_jit_reuses_one_specialization() {
        fn trace<'c>(
            extent: DimensionVariable,
            inputs: Vec<CompilationTracer<Domain<'c>>>,
        ) -> Result<CompilationTracer<Domain<'c>>, Error> {
            let [size, value] = inputs.as_slice() else {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
            };
            let dimension = size.to_dimension(extent)?;
            let broadcast = size
                .context()
                .bind(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), &[value.clone(), dimension])?
                .remove(0);
            Ok(size
                .context()
                .bind(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), &[broadcast])?
                .remove(0))
        }

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let scalar_i64 = replicated_scalar_type(&mesh, DataType::I64);
        let scalar_f32 = replicated_scalar_type(&mesh, DataType::F32);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let function: CompiledFunctionDispatcher<Domain<'_>, (), Vec<ArrayIrType>, ArrayIrType, _> =
            domain.jit_with_options(move |(), inputs| trace(extent.clone(), inputs), Options::new(mesh.clone()));

        let call = |size: i64| {
            let size =
                Array::from_host_buffer(&domain, scalar_i64.clone(), mesh.clone(), size.to_ne_bytes().as_slice())
                    .unwrap();
            let value =
                Array::from_host_buffer(&domain, scalar_f32.clone(), mesh.clone(), 2.0f32.to_ne_bytes().as_slice())
                    .unwrap();
            function.execute((), vec![Value::Array(size), Value::Array(value)]).unwrap()
        };

        assert_eq!(read_f32s(&client, program_array(&call(2))), vec![4.0]);
        assert_eq!(read_f32s(&client, program_array(&call(4))), vec![8.0]);
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_misses, 1);
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.traces, 1);
        assert_eq!(statistics.lowerings, 1);
        assert_eq!(statistics.compilation_requests, 1);
        assert_eq!(function.executable_specialization_count(), 1);
        assert_eq!(domain.program_count(), 1);
    }

    #[test]
    fn test_data_derived_padding_disciplines_execute_without_observable_padding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let program = data_derived_padding_fixture(size_type.clone(), extent);

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let execute = |size: i64| {
            let input =
                Array::from_host_buffer(&domain, size_type.clone(), mesh.clone(), size.to_ne_bytes().as_slice())
                    .unwrap();
            let eager = program
                .interpret_in_context(&domain, vec![Value::Array(input.clone())])
                .unwrap()
                .into_iter()
                .map(|value| match value {
                    Value::Array(array) => array,
                    Value::Dimension(_) => panic!("padding fixture returned a first-class dimension"),
                    Value::Reference(_) => panic!("padding fixture returned a reference"),
                })
                .collect::<Vec<_>>();
            let compiled = domain.execute_program(&compiled, vec![input]).unwrap();
            assert_eq!(compiled.len(), eager.len());
            for (compiled, eager) in compiled.iter().zip(&eager) {
                assert_eq!(compiled.shape(), eager.shape());
                assert_eq!(compiled.data_type(), eager.data_type());
                let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
                let compiled_bytes = compiled
                    .device_shard(device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                let eager_bytes = eager
                    .device_shard(device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                assert_eq!(compiled_bytes, eager_bytes);
            }
            compiled
        };

        let small = execute(2);
        assert_eq!(small[0].shape(), StaticShape::new(vec![2]));
        assert_eq!(read_f32s(&client, &small[0]), vec![-1.0, 0.0]);
        assert_eq!(read_f32s(&client, &small[1]), vec![1.0, 0.0]);
        assert_eq!(read_u64s(&client, &small[2]), vec![1, 0]);
        assert_eq!(read_u64s(&client, &small[3]), vec![1, 0]);
        assert_eq!(read_f32s(&client, &small[4]), vec![1.0]);
        assert_eq!(read_f32s(&client, &small[5]), vec![1.0]);
        assert_eq!(read_f32s(&client, &small[6]), vec![-1.0]);
        assert_eq!(read_booleans(&client, &small[7]), vec![true]);
        assert_eq!(read_booleans(&client, &small[8]), vec![true]);
        assert_eq!(read_f32s(&client, &small[9]), vec![1.0]);
        assert_eq!(read_f32s(&client, &small[12]), vec![2.0]);
        assert_eq!(read_i32s(&client, &small[13]), vec![1]);
        assert_eq!(read_i32s(&client, &small[14]), vec![1]);
        assert_eq!(read_i32s(&client, &small[15]), vec![3, 3]);
        assert_eq!(read_i32s(&client, &small[16]), vec![1, 1, 1, 1]);

        // Row `i` of the `i + j` matrix reduces to `i + ln(1 + e + e² + e³)`, so exactly the live rows appear and a
        // physical suffix row would show up as an extra element. The tolerance is here because the expansion runs
        // `exp` and `log`, not because the discipline is approximate: the byte comparison against the reference
        // kernel inside `execute` is exact.
        assert_eq!(small[10].shape(), StaticShape::new(vec![2]));
        let log_sum_exp = read_f32s(&client, &small[10]);
        assert!((log_sum_exp[0] - 3.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");
        assert!((log_sum_exp[1] - 4.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");

        // The prefix scan runs along the static axis, so row `i` accumulates `[i, 2i + 1, 3i + 3, 4i + 6]` while the
        // data-derived extent rides through on the rows it never mixes.
        assert_eq!(small[11].shape(), StaticShape::new(vec![2, 4]));
        assert_eq!(read_f32s(&client, &small[11]), vec![0.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 10.0]);

        // With a single live lane, every physical suffix lane of the data-derived axis is padding, which index
        // reductions must still never select.
        let single = execute(1);
        assert_eq!(read_i32s(&client, &single[13]), vec![0]);
        assert_eq!(read_i32s(&client, &single[14]), vec![0]);
        assert_eq!(read_i32s(&client, &single[15]), vec![3]);
        assert_eq!(read_i32s(&client, &single[16]), vec![0, 0, 0, 0]);

        let at_bound = execute(4);
        assert_eq!(at_bound[0].shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, &at_bound[0]), vec![-3.0, -2.0, -1.0, 0.0]);
        assert_eq!(read_f32s(&client, &at_bound[1]), vec![3.0, 2.0, 1.0, 0.0]);
        assert_eq!(read_u64s(&client, &at_bound[2]), vec![3, 2, 1, 0]);
        assert_eq!(read_u64s(&client, &at_bound[3]), vec![3, 2, 1, 0]);
        assert_eq!(read_f32s(&client, &at_bound[4]), vec![6.0]);
        assert_eq!(read_f32s(&client, &at_bound[5]), vec![3.0]);
        assert_eq!(read_f32s(&client, &at_bound[6]), vec![-3.0]);
        assert_eq!(read_booleans(&client, &at_bound[7]), vec![true]);
        assert_eq!(read_booleans(&client, &at_bound[8]), vec![true]);
        assert_eq!(read_f32s(&client, &at_bound[9]), vec![14.0]);
        assert_eq!(read_f32s(&client, &at_bound[12]), vec![24.0]);
        assert_eq!(read_i32s(&client, &at_bound[13]), vec![3]);
        assert_eq!(read_i32s(&client, &at_bound[14]), vec![3]);
        assert_eq!(read_i32s(&client, &at_bound[15]), vec![3, 3, 3, 3]);
        assert_eq!(read_i32s(&client, &at_bound[16]), vec![3, 3, 3, 3]);

        // At the bound the reduction and the scan grow by exactly the two extra live rows, which is what rules out a
        // padded row silently participating at the smaller size.
        assert_eq!(at_bound[10].shape(), StaticShape::new(vec![4]));
        let log_sum_exp = read_f32s(&client, &at_bound[10]);
        assert!((log_sum_exp[0] - 3.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");
        assert!((log_sum_exp[1] - 4.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");
        assert!((log_sum_exp[2] - 5.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");
        assert!((log_sum_exp[3] - 6.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");
        assert_eq!(at_bound[11].shape(), StaticShape::new(vec![4, 4]));
        assert_eq!(
            read_f32s(&client, &at_bound[11]),
            vec![0.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 10.0, 2.0, 5.0, 9.0, 14.0, 3.0, 7.0, 12.0, 18.0],
        );
    }

    #[test]
    fn test_data_derived_scaled_dot_and_attention_execute_without_observable_padding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let program = data_derived_scaled_dot_attention_fixture(
            size_type.clone(),
            extent,
            DataType::F32,
            DataType::F32,
            AttentionImplementation::Automatic,
        );
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert!(lowered.stable_hlo().contains("stablehlo.composite \"xla.scaled_dot\""));
        assert!(!lowered.stable_hlo().contains("__cudnn$fmha"));
        let compiled = domain.compile_program(&lowered).unwrap();
        let execute = |size: i64| {
            let input =
                Array::from_host_buffer(&domain, size_type.clone(), mesh.clone(), size.to_ne_bytes().as_slice())
                    .unwrap();
            let eager = program
                .interpret_in_context(&domain, vec![Value::Array(input.clone())])
                .unwrap()
                .into_iter()
                .map(|value| match value {
                    Value::Array(array) => array,
                    Value::Dimension(_) => panic!("attention fixture returned a first-class dimension"),
                    Value::Reference(_) => panic!("attention fixture returned a reference"),
                })
                .collect::<Vec<_>>();
            let compiled = domain.execute_program(&compiled, vec![input]).unwrap();
            assert_eq!(compiled.len(), eager.len());
            for (compiled, eager) in compiled.iter().zip(&eager) {
                assert_eq!(compiled.shape(), eager.shape());
                assert_eq!(compiled.data_type(), eager.data_type());
                let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
                assert_eq!(
                    compiled
                        .device_shard(device_id)
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .copy_to_host(None)
                        .unwrap()
                        .r#await()
                        .unwrap(),
                    eager
                        .device_shard(device_id)
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .copy_to_host(None)
                        .unwrap()
                        .r#await()
                        .unwrap(),
                );
            }
            compiled
        };

        let small = execute(2);
        assert_eq!(small[0].shape(), StaticShape::new(vec![2, 1]));
        assert_eq!(read_f32s(&client, &small[0]), vec![16.0; 2]);
        assert_eq!(small[1].shape(), StaticShape::new(vec![2, 2, 1, 8]));
        assert_eq!(read_bf16s(&client, &small[1]), vec![0.0; 32]);
        assert_eq!(read_bf16s(&client, &small[2]), vec![0.0; 32]);
        assert_eq!(read_bf16s(&client, &small[3]), vec![0.0; 32]);
        assert_eq!(
            read_bf16s(&client, &small[4]),
            [1.0, 0.0].into_iter().cycle().take(4).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
        );

        let at_bound = execute(4);
        assert_eq!(at_bound[0].shape(), StaticShape::new(vec![4, 1]));
        assert_eq!(read_f32s(&client, &at_bound[0]), vec![16.0; 4]);
        assert_eq!(at_bound[1].shape(), StaticShape::new(vec![4, 4, 1, 8]));
        assert_eq!(
            read_bf16s(&client, &at_bound[1]),
            [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
        );
        assert_eq!(read_bf16s(&client, &at_bound[2]), vec![0.0; 128]);
        // Three visible keys have equal scores and values 0, 1, 2. Each query contributes
        // `dS[j] = (1 / 3) * 8 * (j - 1)`, so summing three visible queries gives `dK[j] = 8 * (j - 1)`
        // and `dV[j] = 3 * (1 / 3) = 1`. The rounded BF16 residual must not rescale these probabilities.
        assert_eq!(
            read_bf16s(&client, &at_bound[3]),
            [-8.0, 0.0, 8.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
        );
        assert_eq!(
            read_bf16s(&client, &at_bound[4]),
            [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
        );

        let zero =
            Array::from_host_buffer(&domain, size_type.clone(), mesh.clone(), 0i64.to_ne_bytes().as_slice()).unwrap();
        let error = domain.execute_program(&compiled, vec![zero]).unwrap_err();
        assert!(
            error.to_string().contains(
                "`dimension_from_scalar` failed: input dimension `extent` = 0 is outside its declared bounds [1, 5)"
            ),
            "{error}",
        );

        // Eager execution replays the attention calls of the bounded-dynamic program at the exact types of their
        // inputs, so a program that explicitly requests the fused kernels reports that it cannot use them instead of
        // silently computing the portable result.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let fused = data_derived_scaled_dot_attention_fixture(
            size_type.clone(),
            extent,
            DataType::F32,
            DataType::F32,
            AttentionImplementation::Fused,
        );
        let input = Array::from_host_buffer(&domain, size_type, mesh, 2i64.to_ne_bytes().as_slice()).unwrap();
        let error = fused.interpret_in_context(&domain, vec![Value::Array(input)]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("the fused attention implementation is unavailable because the target platform is not CUDA"),
            "{error}",
        );
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_cuda_data_derived_padding_disciplines_execute_without_observable_padding() {
        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let program = data_derived_padding_fixture(size_type.clone(), extent);
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let execute = |size: i64| {
            let input =
                Array::from_host_buffer(&domain, size_type.clone(), mesh.clone(), size.to_ne_bytes().as_slice())
                    .unwrap();
            domain.execute_program(&compiled, vec![input]).unwrap()
        };

        let small = execute(2);
        assert_eq!(read_f32s(&client, &small[0]), vec![-1.0, 0.0]);
        assert_eq!(read_f32s(&client, &small[1]), vec![1.0, 0.0]);
        assert_eq!(read_u64s(&client, &small[2]), vec![1, 0]);
        assert_eq!(read_u64s(&client, &small[3]), vec![1, 0]);
        assert_eq!(read_f32s(&client, &small[4]), vec![1.0]);
        assert_eq!(read_f32s(&client, &small[5]), vec![1.0]);
        assert_eq!(read_f32s(&client, &small[6]), vec![-1.0]);
        assert_eq!(read_booleans(&client, &small[7]), vec![true]);
        assert_eq!(read_booleans(&client, &small[8]), vec![true]);
        assert_eq!(read_f32s(&client, &small[9]), vec![1.0]);
        assert_eq!(read_f32s(&client, &small[12]), vec![2.0]);
        assert_eq!(read_i32s(&client, &small[13]), vec![1]);
        assert_eq!(read_i32s(&client, &small[14]), vec![1]);
        assert_eq!(read_i32s(&client, &small[15]), vec![3, 3]);
        assert_eq!(read_i32s(&client, &small[16]), vec![1, 1, 1, 1]);

        let at_bound = execute(4);
        assert_eq!(read_f32s(&client, &at_bound[0]), vec![-3.0, -2.0, -1.0, 0.0]);
        assert_eq!(read_f32s(&client, &at_bound[1]), vec![3.0, 2.0, 1.0, 0.0]);
        assert_eq!(read_u64s(&client, &at_bound[2]), vec![3, 2, 1, 0]);
        assert_eq!(read_u64s(&client, &at_bound[3]), vec![3, 2, 1, 0]);
        assert_eq!(read_f32s(&client, &at_bound[4]), vec![6.0]);
        assert_eq!(read_f32s(&client, &at_bound[5]), vec![3.0]);
        assert_eq!(read_f32s(&client, &at_bound[6]), vec![-3.0]);
        assert_eq!(read_booleans(&client, &at_bound[7]), vec![true]);
        assert_eq!(read_booleans(&client, &at_bound[8]), vec![true]);
        assert_eq!(read_f32s(&client, &at_bound[9]), vec![14.0]);
        assert_eq!(read_f32s(&client, &at_bound[12]), vec![24.0]);
        assert_eq!(read_i32s(&client, &at_bound[13]), vec![3]);
        assert_eq!(read_i32s(&client, &at_bound[14]), vec![3]);
        assert_eq!(read_i32s(&client, &at_bound[15]), vec![3, 3, 3, 3]);
        assert_eq!(read_i32s(&client, &at_bound[16]), vec![3, 3, 3, 3]);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_cuda_data_derived_scaled_dot_and_attention_execute_fused_without_observable_padding() {
        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        for (scaled_element_type, scaled_scale_type) in [
            (DataType::F4E2M1FN, DataType::F8E4M3FN),
            (DataType::F8E4M3FN, DataType::F8E8M0FNU),
            (DataType::F8E5M2, DataType::F8E8M0FNU),
        ] {
            let scaled_dot_value = match scaled_element_type {
                DataType::F8E4M3FN | DataType::F8E5M2 => 32.0,
                _ => 16.0,
            };
            let size_type = replicated_scalar_type(&mesh, DataType::I64);
            let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
            let program = data_derived_scaled_dot_attention_fixture(
                size_type.clone(),
                extent,
                scaled_element_type,
                scaled_scale_type,
                AttentionImplementation::Fused,
            );
            let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
            // cuDNN accepts the NVFP4 and `f8e4m3fn` MXFP8 pairs, which lower to the block-scaled-dot custom call, but
            // not `f8e5m2` on both sides, which keeps the composite.
            let uses_custom_call = scaled_element_type != DataType::F8E5M2;
            assert_eq!(lowered.stable_hlo().contains("stablehlo.custom_call @__op$block_scaled_dot"), uses_custom_call);
            assert_eq!(lowered.stable_hlo().contains("stablehlo.composite \"xla.scaled_dot\""), !uses_custom_call);
            assert!(lowered.stable_hlo().contains("__cudnn$fmhaSoftmax"));
            assert!(lowered.stable_hlo().contains("__cudnn$fmhaSoftmaxBackward"));
            let compiled = domain.compile_program(&lowered).unwrap();
            let execute = |size: i64| {
                let input =
                    Array::from_host_buffer(&domain, size_type.clone(), mesh.clone(), size.to_ne_bytes().as_slice())
                        .unwrap();
                domain.execute_program(&compiled, vec![input]).unwrap()
            };

            let small = execute(2);
            assert_eq!(small[0].shape(), StaticShape::new(vec![2, 1]));
            assert_eq!(read_f32s(&client, &small[0]), vec![scaled_dot_value; 2]);
            assert_eq!(small[1].shape(), StaticShape::new(vec![2, 2, 1, 8]));
            assert_eq!(read_bf16s(&client, &small[1]), vec![0.0; 32]);
            assert_eq!(read_bf16s(&client, &small[2]), vec![0.0; 32]);
            assert_eq!(read_bf16s(&client, &small[3]), vec![0.0; 32]);
            assert_eq!(
                read_bf16s(&client, &small[4]),
                [1.0, 0.0].into_iter().cycle().take(4).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
            );

            let at_bound = execute(4);
            assert_eq!(at_bound[0].shape(), StaticShape::new(vec![4, 1]));
            assert_eq!(read_f32s(&client, &at_bound[0]), vec![scaled_dot_value; 4]);
            assert_eq!(at_bound[1].shape(), StaticShape::new(vec![4, 4, 1, 8]));
            assert_eq!(
                read_bf16s(&client, &at_bound[1]),
                [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
            );
            assert_eq!(read_bf16s(&client, &at_bound[2]), vec![0.0; 128]);
            let actual_query_gradient = read_bf16s(&client, &at_bound[3]);
            let expected_query_gradient =
                [-8.0, 0.0, 8.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>();
            // cuDNN's fused BF16 backward-attention kernel is not bit-identical to the unfused reference
            // composition, so compare its gradients with an explicit reduced-precision tolerance.
            actual_query_gradient.iter().zip(&expected_query_gradient).for_each(|(actual, expected)| {
                assert!((actual - expected).abs() <= 0.25, "expected {expected} within 0.25, but found {actual}");
            });
            let actual_value_gradient = read_bf16s(&client, &at_bound[4]);
            let expected_value_gradient =
                [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>();
            actual_value_gradient.iter().zip(&expected_value_gradient).for_each(|(actual, expected)| {
                assert!((actual - expected).abs() <= 0.25, "expected {expected} within 0.25, but found {actual}");
            });

            let zero =
                Array::from_host_buffer(&domain, size_type, mesh.clone(), 0i64.to_ne_bytes().as_slice()).unwrap();
            let error = domain.execute_program(&compiled, vec![zero]).unwrap_err();
            assert!(
                error.to_string().contains(
                    "`dimension_from_scalar` failed: input dimension `extent` = 0 is outside its declared bounds [1, 5)"
                ),
                "{error}",
            );
        }
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "hardware throughput benchmark; run explicitly on a CUDA system"]
    fn test_cuda_fused_attention_throughput() {
        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let [batch, sequence, heads, head_dimension] = [4, 2048, 32, 128];
        let r#type = ArrayType::new(
            DataType::BF16,
            Shape::new(vec![
                CoreDimension::Static(batch),
                CoreDimension::Static(sequence),
                CoreDimension::Static(heads),
                CoreDimension::Static(head_dimension),
            ]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 4))
        .unwrap();
        let attention = dot_product_attention::<Domain<'static>>(
            AttentionConfiguration::new().with_scale(1.0 / (head_dimension as f64).sqrt()),
        );
        let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
                let (output, _) =
                    attention.call(AttentionInputs::new(inputs[0].clone(), inputs[1].clone(), inputs[2].clone()))?;
                Ok(vec![output])
            },
            vec![ArrayIrType::from(r#type.clone()); 3],
        )
        .unwrap();
        let program = program.into_flat_program();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert!(lowered.stable_hlo().contains("__cudnn$fmhaSoftmax"));
        let compilation_start = Instant::now();
        let compiled = domain.compile_program(&lowered).unwrap();
        let compilation_duration = compilation_start.elapsed();

        let one = 0x3f80u16.to_ne_bytes();
        let mut bytes = vec![0u8; batch * sequence * heads * head_dimension * size_of::<u16>()];
        bytes.chunks_exact_mut(size_of::<u16>()).for_each(|element| element.copy_from_slice(&one));
        let input = Array::from_host_buffer(&domain, r#type, mesh, bytes.as_slice()).unwrap();
        input.block_until_ready().unwrap();
        let execute = || {
            let outputs = domain.execute_program(&compiled, vec![input.clone(), input.clone(), input.clone()])?;
            outputs[0].block_until_ready()?;
            Ok::<_, Error>(())
        };
        // The Spark idles at its lowest power state, so warm the fused kernel long enough for clocks to stabilize
        // before collecting synchronized samples.
        (0..20).for_each(|_| execute().unwrap());
        let mut samples = (0..20)
            .map(|_| {
                let start = Instant::now();
                execute().unwrap();
                start.elapsed().as_nanos()
            })
            .collect::<Vec<_>>();
        samples.sort_unstable();
        let median_nanoseconds = samples[samples.len() / 2];
        let floating_point_operations =
            4.0 * batch as f64 * heads as f64 * sequence as f64 * sequence as f64 * head_dimension as f64;
        let throughput_teraflops = floating_point_operations / (median_nanoseconds as f64 * 1e3);

        eprintln!(
            "fused attention b{batch} h{heads} s{sequence} d{head_dimension}: compile={compilation_duration:?}, \
             median={:.3} ms, throughput={throughput_teraflops:.2} TFLOP/s",
            median_nanoseconds as f64 / 1e6,
        );
    }
}
