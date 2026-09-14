//! XLA staging carriers for portable kernels with ordinary attached computation regions.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use ryft_core::kernels::{
    KernelCallOperation, KernelCompiler, KernelDefinition, KernelOperation, KernelSchedule, VerifiedKernel,
};
use ryft_core::{
    Array as CpuArray, ArrayIrType, ArrayIrValue, Atom, AtomId, ConstantOperation, Context, CotangentAccumulator,
    DifferentiableOperation, DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    DifferentiationPolicy, Domain, Effects, InputRegionProvenance, Instruction, InterpretableOperation,
    InterpretationDriver, MaybeZero, Operation, OutputRegionProvenance, PartialValue, PartiallyEvaluatableOperation,
    Placeholder, Program, ProgramError, ReferenceAccessMode, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargePolicy, ReferenceDischargeValue, ReferenceDischargeableOperation, Region, RegionInterface,
    RegionSlot, Tracer, TracingContext, TransposableOperation, TranspositionContext, TranspositionDriver, Type,
    TypeError, TypeIdentityRenaming, Typed, Value,
};

use crate::experimental::ops::{FlatXlaProgram, XlaConstant, XlaOperation};
use crate::kernels::{CompiledKernel, KernelEmbeddingError, KernelOutputEmbedding};

/// Canonical kernel instruction retained inside the XLA operation family until explicit compiler selection.
///
/// The payload contains only operation metadata. Its computations remain ordinary attached regions of the XLA
/// program, so effect, reference, identity, and structural validation continue to inspect the actual current body.
/// Generic differentiation and reference discharge require an owner-supported kernel rule and reject this carrier.
#[derive(Clone, Debug)]
pub struct XlaKernelOperation(pub(crate) KernelOperation);

impl XlaKernelOperation {
    /// Wraps a canonical portable instruction without changing its semantic contract.
    pub fn new(operation: KernelOperation) -> Self {
        Self(operation)
    }

    /// Returns the canonical operation metadata; computations belong to the containing instruction's regions.
    pub fn operation(&self) -> &KernelOperation {
        &self.0
    }
}

impl Display for XlaKernelOperation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, formatter)
    }
}

impl Operation for XlaKernelOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn region_slots(&self) -> &'static [RegionSlot] {
        self.0.region_slots()
    }

    fn infer_region_input_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
        self.0.infer_region_input_types(input_types, region_interfaces)
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        self.0.infer_output_types(input_types, region_interfaces)
    }

    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        self.0.input_region_provenance(region_index, input_index)
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        self.0.output_region_provenance(output_index)
    }

    fn is_zero(&self, output_index: usize) -> bool {
        self.0.is_zero(output_index)
    }

    fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
        self.0.region_capture_input_count(region_index)
    }

    fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
        self.0.reference_output_identity_input(output_index)
    }

    fn allows_reference_access_through_region_input(&self, region_index: usize, mode: ReferenceAccessMode) -> bool {
        self.0.allows_reference_access_through_region_input(region_index, mode)
    }

    fn effects(&self) -> Cow<'_, Effects> {
        self.0.effects()
    }

    fn render(&self, formatter: &mut Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.0.render(formatter, indentation)
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayIrType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self(self.0.rename_type_identities(renaming)?))
    }
}

impl<C: Context<Type = ArrayIrType>, P: ReferenceDischargePolicy<C>> ReferenceDischargeableOperation<C, P>
    for XlaKernelOperation
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        _context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        _inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        Err(ProgramError::MalformedProgram(format!(
            "unselected kernel operation `{}` requires owner reference discharge",
            self.name(),
        )))
    }
}

impl<C: Domain> InterpretableOperation<C> for XlaKernelOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Err(ProgramError::MalformedProgram(format!(
            "unselected kernel operation `{}` requires XLA compiler selection before interpretation",
            self.name(),
        )))
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for XlaKernelOperation where Self: Into<C::Operation> {}

impl<C: Context<Type = ArrayIrType>> DifferentiableOperation<C> for XlaKernelOperation {
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        _context: &DifferentiationContext<C, P>,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::MalformedProgram(format!(
            "unselected kernel operation `{}` has no differentiation rule",
            self.name(),
        ))
        .into())
    }
}

impl<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>> TransposableOperation<V, O>
    for XlaKernelOperation
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TranspositionContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        _accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        Err(ProgramError::MalformedProgram(format!(
            "unselected kernel operation `{}` has no transposition rule",
            self.name(),
        ))
        .into())
    }
}

/// Converts a verified definition's body into the XLA value universe without hiding its region graph.
///
/// This is a whole-arena structural conversion, like `Program::map_operations`: atom and region identifiers stay
/// stable, including shared descendants. Array literal atoms become variables defined by existing constant
/// instructions at the region entrance. No runtime buffers, captures, or references become literal XLA values.
fn stage_body(definition: &KernelDefinition) -> Result<FlatXlaProgram, ProgramError> {
    let body = definition.body();
    let regions = body
        .regions()
        .iter()
        .map(|region| {
            let mut instructions = Vec::new();
            let atoms = region
                .atoms()
                .iter()
                .enumerate()
                .map(|(index, atom)| {
                    Ok(match atom {
                        Atom::Variable(r#type) => Atom::Variable(r#type.clone()),
                        Atom::Constant(ArrayIrValue::Dimension(value)) => {
                            Atom::Constant(XlaConstant::Dimension(value.clone()))
                        }
                        Atom::Constant(ArrayIrValue::Array(value)) => {
                            instructions.push(Instruction::new(
                                ConstantOperation::new(value.clone()).into(),
                                vec![],
                                vec![AtomId::new(index)],
                                vec![],
                            ));
                            Atom::Variable(ArrayIrType::Array(value.r#type().into_owned()))
                        }
                        Atom::Constant(ArrayIrValue::Reference(_)) => {
                            return Err(ProgramError::MalformedProgram(
                                "kernel body cannot capture a literal reference".to_owned(),
                            ));
                        }
                    })
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            instructions.extend(region.instructions().iter().map(|instruction| {
                Instruction::new(
                    XlaOperation::Kernel(XlaKernelOperation::new(instruction.operation().clone())),
                    instruction.inputs().to_vec(),
                    instruction.outputs().to_vec(),
                    instruction.regions().to_vec(),
                )
                .with_provenance(instruction.provenance().clone())
            }));
            Ok(Region::new(atoms, region.input_ids().to_vec(), region.output_ids().to_vec(), instructions))
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    Program::new(vec![Placeholder; body.input_count()], vec![Placeholder; body.output_count()], regions, body.entry())
}

/// Binds an unselected portable kernel with its real body attached to the surrounding XLA program.
///
/// Compiler selection belongs to the enclosing XLA domain's immutable compilation options. Staging itself neither
/// invokes an adapter nor acquires native runtime resources.
pub fn stage_kernel<C>(
    context: &C,
    definition: &KernelDefinition,
    inputs: &[C::Value],
) -> Result<Vec<C::Value>, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = XlaConstant, Operation = XlaOperation>,
{
    context.bind(
        XlaOperation::Kernel(XlaKernelOperation::new(KernelOperation::Call(definition.operation().clone()))),
        vec![stage_body(definition)?],
        inputs,
    )
}

/// Restores the current attached body for the core verifier; retained operation metadata is never a body proof.
fn definition_from_body(
    operation: &KernelCallOperation,
    body: ryft_core::RegionRef<'_, XlaConstant, XlaOperation>,
) -> Result<KernelDefinition, ProgramError> {
    let body = body.to_program();
    let regions = body
        .regions()
        .iter()
        .map(|region| {
            let atoms = region
                .atoms()
                .iter()
                .map(|atom| {
                    Ok(match atom {
                        Atom::Variable(r#type) => Atom::Variable(r#type.clone()),
                        Atom::Constant(XlaConstant::Dimension(value)) => {
                            Atom::Constant(ArrayIrValue::<CpuArray>::Dimension(value.clone()))
                        }
                        Atom::Constant(XlaConstant::Captured(_)) => {
                            return Err(ProgramError::MalformedProgram(
                                "kernel body cannot capture an external XLA value".to_owned(),
                            ));
                        }
                    })
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let instructions = region
                .instructions()
                .iter()
                .map(|instruction| {
                    let operation = match instruction.operation() {
                        XlaOperation::Kernel(operation) => operation.0.clone(),
                        XlaOperation::Array(ryft_core::ArrayOperation::Constant(operation)) => {
                            KernelOperation::Portable(ryft_core::ArrayOperation::Constant(operation.clone()).into())
                        }
                        operation => {
                            return Err(ProgramError::MalformedProgram(format!(
                                "XLA operation `{}` has no portable kernel reconstruction",
                                operation.name(),
                            )));
                        }
                    };
                    Ok(Instruction::new(
                        operation,
                        instruction.inputs().to_vec(),
                        instruction.outputs().to_vec(),
                        instruction.regions().to_vec(),
                    )
                    .with_provenance(instruction.provenance().clone()))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            Ok(Region::new(atoms, region.input_ids().to_vec(), region.output_ids().to_vec(), instructions))
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let body = Program::new(
        vec![Placeholder; body.input_count()],
        vec![Placeholder; body.output_count()],
        regions,
        body.entry(),
    )?;
    KernelDefinition::new(operation.clone(), body).map_err(|error| ProgramError::MalformedProgram(error.to_string()))
}

/// PJRT facts for one selected execution device, in the mesh's physical order.
#[derive(Clone, Debug)]
pub struct XlaKernelDeviceFacts {
    /// PJRT device kind, including the implementation's architecture description.
    pub kind: String,

    /// Typed plugin attributes, including compute capability when the plugin exposes it.
    pub attributes: BTreeMap<String, ryft_pjrt::Value>,
}

/// Actual execution facts joined with the adapter target before kernel admission and compilation.
///
/// These facts come from the live PJRT client and selected mesh, rather than caller-provided target claims. They
/// enter the selected custom call's configuration identity before the ordinary lowered-program cache lookup.
#[derive(Clone, Debug)]
pub struct XlaKernelExecutionFacts {
    /// Live PJRT platform name.
    pub platform_name: String,

    /// Live PJRT platform implementation version.
    pub platform_version: String,

    /// Live plugin's PJRT C API version.
    pub pjrt_version: ryft_pjrt::Version,

    /// Whether the live plugin exposes the PJRT FFI extension.
    pub has_ffi_extension: bool,

    /// Live client attributes, in deterministic name order.
    pub attributes: BTreeMap<String, ryft_pjrt::Value>,

    /// Actual selected devices, preserving mesh order and multiplicity.
    pub devices: Vec<XlaKernelDeviceFacts>,
}

impl XlaKernelExecutionFacts {
    /// Reads target facts from the live execution owner without loading an additional device runtime.
    pub fn from_client(
        client: &ryft_pjrt::Client<'_>,
        mesh: &ryft_core::DeviceMesh,
    ) -> Result<Self, KernelEmbeddingError> {
        let devices = client.devices()?;
        let devices = mesh
            .devices()
            .iter()
            .map(|mesh_device| {
                let device = devices
                    .iter()
                    .find(|device| device.id().is_ok_and(|id| id == mesh_device.id()))
                    .ok_or_else(|| KernelEmbeddingError::Invalid {
                        message: format!(
                            "kernel execution device {} is not visible to the live client",
                            mesh_device.id()
                        ),
                    })?;
                Ok(XlaKernelDeviceFacts {
                    kind: device.kind()?.into_owned(),
                    attributes: device
                        .attributes()?
                        .iter()
                        .map(|(name, value)| (name.clone(), value.clone()))
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>, KernelEmbeddingError>>()?;
        Ok(Self {
            platform_name: client.platform_name()?.into_owned(),
            platform_version: client.platform_version()?.into_owned(),
            pjrt_version: client.version(),
            has_ffi_extension: client.ffi_extension().is_ok(),
            attributes: client.attributes()?.iter().map(|(name, value)| (name.clone(), value.clone())).collect(),
            devices,
        })
    }

    /// Encodes every fact with explicit variant tags, lengths, and exact floating-point bits.
    pub(crate) fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
        /// Encodes the finite PJRT attribute family without lossy floating-point formatting.
        fn attributes(values: &BTreeMap<String, ryft_pjrt::Value>) -> serde_json::Value {
            serde_json::Value::Array(
                values
                    .iter()
                    .map(|(name, value)| {
                        let value = match value {
                            ryft_pjrt::Value::Bool(value) => serde_json::json!(["bool", value]),
                            ryft_pjrt::Value::I64(value) => serde_json::json!(["i64", value]),
                            ryft_pjrt::Value::I64List(value) => serde_json::json!(["i64_list", value]),
                            ryft_pjrt::Value::F32(value) => serde_json::json!(["f32", value.to_bits()]),
                            ryft_pjrt::Value::String(value) => serde_json::json!(["string", value]),
                        };
                        serde_json::json!([name, value])
                    })
                    .collect(),
            )
        }
        Ok(serde_json::to_vec(&serde_json::json!([
            1,
            self.platform_name,
            self.platform_version,
            self.pjrt_version.major,
            self.pjrt_version.minor,
            self.has_ffi_extension,
            attributes(&self.attributes),
            self.devices
                .iter()
                .map(|device| serde_json::json!([device.kind, attributes(&device.attributes)]))
                .collect::<Vec<_>>()
        ]))?)
    }
}

/// XLA integration contract joining an adapter-owned target with actual PJRT execution facts.
///
/// Implementations belong in the integration that knows both systems; adapter crates need not depend on XLA.
/// Reject unsupported platforms, device architectures, and plugin ABI requirements before adapter admission.
/// A successful check grants no fallback, implicit target substitution, or unchecked runtime feature support.
pub trait XlaKernelTarget {
    /// Validates this immutable compilation target against all actual selected execution devices and the plugin ABI.
    fn admit_execution(&self, facts: &XlaKernelExecutionFacts) -> Result<(), KernelEmbeddingError>;
}

/// One explicitly enabled compiler and typed embedding selected by the enclosing XLA compilation options.
///
/// The binding is immutable and uses a complete configuration key for equality and hashing. It does not own an
/// executable, runtime registry, or cache. Each selection revalidates the current attached body before invoking the
/// captured typed adapter. There is no fallback to another compiler or host interpretation.
#[derive(Clone)]
pub struct XlaKernelCompilerBinding {
    /// Complete adapter and embedding identity, with unambiguous component lengths.
    configuration: Vec<u8>,

    /// Host verifier work limit; this affects admission work, not compiled kernel semantics.
    maximum_programs: usize,

    /// Typed adapter and embedding retained behind the XLA-owned selection boundary.
    compile: Arc<
        dyn Fn(&KernelDefinition, &XlaKernelExecutionFacts) -> Result<CompiledKernel, KernelEmbeddingError>
            + Send
            + Sync,
    >,
}

impl XlaKernelCompilerBinding {
    /// Binds a typed adapter and embedding with their immutable target, options, and schedule.
    ///
    /// The embedding supplies every integration choice through its configuration key, including handler target,
    /// physical mapping, payload schema, and plugin ABI requirements. `maximum_programs` bounds core verification work.
    pub fn new<C, B>(
        compiler: C,
        target: C::Target,
        options: C::Options,
        schedule: KernelSchedule,
        embedding: B,
        maximum_programs: usize,
    ) -> Result<Self, KernelEmbeddingError>
    where
        C: 'static + Send + Sync + KernelCompiler<Error: 'static + Send + Sync>,
        C::Target: 'static + Send + Sync + XlaKernelTarget,
        C::Options: 'static + Send + Sync,
        B: 'static + Send + Sync + KernelOutputEmbedding<C::Output>,
    {
        let adapter_configuration = compiler
            .configuration_key(&target, &options, &schedule)
            .map_err(|error| KernelEmbeddingError::Compiler(Box::new(error)))?;
        let embedding_configuration = embedding.configuration_key()?;
        let mut configuration = Vec::new();
        for component in [&adapter_configuration, &embedding_configuration] {
            configuration.extend_from_slice(&(component.len() as u64).to_le_bytes());
            configuration.extend_from_slice(component);
        }
        let output_configuration = configuration.clone();
        let compile = Arc::new(move |definition: &KernelDefinition, facts: &XlaKernelExecutionFacts| {
            target.admit_execution(facts)?;
            let current_configuration = compiler
                .configuration_key(&target, &options, &schedule)
                .map_err(|error| KernelEmbeddingError::Compiler(Box::new(error)))?;
            if current_configuration != adapter_configuration {
                return Err(KernelEmbeddingError::Invalid {
                    message: "selected kernel compiler configuration changed after binding".to_owned(),
                });
            }
            if embedding.configuration_key()? != embedding_configuration {
                return Err(KernelEmbeddingError::Invalid {
                    message: "selected kernel embedding configuration changed after binding".to_owned(),
                });
            }
            let verified = VerifiedKernel::new(definition, maximum_programs)
                .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
            let output = verified
                .compile(&compiler, &target, &options, &schedule)
                .map_err(|error| KernelEmbeddingError::Compiler(Box::new(error)))?;
            let facts_configuration = facts.configuration_key()?;
            let mut configuration = output_configuration.clone();
            configuration.extend_from_slice(&(facts_configuration.len() as u64).to_le_bytes());
            configuration.extend_from_slice(&facts_configuration);
            CompiledKernel::from_output(&verified, &configuration, &output, &embedding)
        });
        Ok(Self { configuration, maximum_programs, compile })
    }

    /// Returns the complete deterministic adapter and XLA embedding configuration identity.
    pub fn configuration(&self) -> &[u8] {
        &self.configuration
    }

    /// Compiles the current attached body after core verification and exact selected-adapter admission.
    pub(crate) fn compile(
        &self,
        definition: &KernelDefinition,
        facts: &XlaKernelExecutionFacts,
    ) -> Result<CompiledKernel, KernelEmbeddingError> {
        (self.compile)(definition, facts)
    }
}

impl Debug for XlaKernelCompilerBinding {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("XlaKernelCompilerBinding")
            .field("configuration", &self.configuration)
            .field("maximum_programs", &self.maximum_programs)
            .finish_non_exhaustive()
    }
}

impl PartialEq for XlaKernelCompilerBinding {
    fn eq(&self, other: &Self) -> bool {
        self.configuration == other.configuration && self.maximum_programs == other.maximum_programs
    }
}

impl Eq for XlaKernelCompilerBinding {}

impl std::hash::Hash for XlaKernelCompilerBinding {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.configuration.hash(state);
        self.maximum_programs.hash(state);
    }
}

/// Selects reachable kernel calls, discharges their proven-local state, and extracts the remaining region closure.
/// Dormant transform-rule regions retain their unselected carriers until a transform makes them executable.
pub(crate) fn select_kernels(
    program: &FlatXlaProgram,
    binding: Option<&XlaKernelCompilerBinding>,
    execution_facts: impl FnOnce() -> Result<XlaKernelExecutionFacts, KernelEmbeddingError>,
) -> Result<Option<FlatXlaProgram>, KernelEmbeddingError> {
    // A selected call owns its entire body. Do not separately select descendants that its compiler consumes.
    // Ordinary computation edges remain visible; dormant rule edges are retained without being compiled.
    let mut executable_regions = HashSet::new();
    let mut pending = vec![program.entry()];
    let mut has_kernel = false;
    while let Some(id) = pending.pop() {
        if !executable_regions.insert(id) {
            continue;
        }
        let region = program.region_ref(id).unwrap();
        for instruction in region.instructions() {
            if matches!(instruction.operation(), XlaOperation::Kernel(XlaKernelOperation(KernelOperation::Call(_)))) {
                has_kernel = true;
                continue;
            }
            pending.extend(instruction.regions().iter().copied().enumerate().filter_map(|(index, id)| {
                (instruction.operation().region_role(index) == Some(ryft_core::RegionRole::Computation)).then_some(id)
            }));
        }
    }
    if !has_kernel {
        return Ok(None);
    }
    let binding = binding.ok_or_else(|| KernelEmbeddingError::Invalid {
        message: "kernel call has no explicitly enabled XLA compiler binding".to_owned(),
    })?;
    let facts = execution_facts()?;
    let mut selected = false;
    let regions = program
        .regions()
        .iter()
        .enumerate()
        .map(|(index, region)| {
            let instructions = region
                .instructions()
                .iter()
                .map(|instruction| {
                    if executable_regions.contains(&ryft_core::RegionId::new(index)) {
                        if let XlaOperation::Kernel(XlaKernelOperation(KernelOperation::Call(operation))) =
                            instruction.operation()
                        {
                            let body = ryft_core::RegionRef::new(program.regions(), instruction.regions()[0])
                                .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
                            let definition = definition_from_body(operation, body)
                                .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
                            let compiled = binding.compile(&definition, &facts)?;
                            selected = true;
                            return Ok(Instruction::new(
                                compiled.custom_call().clone().into(),
                                instruction.inputs().to_vec(),
                                instruction.outputs().to_vec(),
                                vec![],
                            )
                            .with_provenance(instruction.provenance().clone()));
                        }
                    }
                    Ok(instruction.clone())
                })
                .collect::<Result<Vec<_>, KernelEmbeddingError>>()?;
            Ok(Region::new(
                region.atoms().to_vec(),
                region.input_ids().to_vec(),
                region.output_ids().to_vec(),
                instructions,
            ))
        })
        .collect::<Result<Vec<_>, KernelEmbeddingError>>()?;
    if !selected {
        return Ok(None);
    }
    let arena = ryft_core::RegionArena::from_regions(regions)
        .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
    let selected = ryft_core::RegionRef::new(&arena, program.entry())
        .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?
        .to_program();
    for region in selected.entry_region_ref().computation_regions() {
        if let Some(instruction) = region
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), XlaOperation::Kernel(_)))
        {
            return Err(KernelEmbeddingError::Invalid {
                message: format!(
                    "kernel operation `{}` remains outside a selected kernel call",
                    instruction.operation().name()
                ),
            });
        }
    }
    Ok(Some(selected))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::EffectClass;
    use ryft_core::kernels::{KernelCompilationError, KernelCompiler, KernelSchedule, VerifiedKernel};
    use ryft_core::operations::custom_call::CustomCallOperation;

    use crate::FromPjrt;
    use crate::kernels::{KernelEmbeddingError, KernelOutputEmbedding};

    use super::*;

    /// Typed fixture compiler that admits only a caller-selected target and retains exact option identity.
    struct Compiler;

    /// Explicit fixture target whose admission checks the actual execution platform first.
    struct FixtureTarget(bool);

    impl XlaKernelTarget for FixtureTarget {
        fn admit_execution(&self, facts: &XlaKernelExecutionFacts) -> Result<(), KernelEmbeddingError> {
            if facts.platform_name != "fixture" {
                return Err(KernelEmbeddingError::Invalid {
                    message: "fixture execution platform mismatch".to_owned(),
                });
            }
            Ok(())
        }
    }

    /// Deterministic execution facts for the hand-authored embedding fixture.
    fn facts() -> Result<XlaKernelExecutionFacts, KernelEmbeddingError> {
        Ok(XlaKernelExecutionFacts {
            platform_name: "fixture".to_owned(),
            platform_version: "1".to_owned(),
            pjrt_version: ryft_pjrt::Version { major: 0, minor: 115 },
            has_ffi_extension: true,
            attributes: Default::default(),
            devices: vec![],
        })
    }

    impl KernelCompiler for Compiler {
        type Target = FixtureTarget;
        type Options = u32;
        type Output = ();
        type Error = std::io::Error;

        fn admit(
            &self,
            _kernel: &VerifiedKernel<'_>,
            target: &FixtureTarget,
            _options: &u32,
            _schedule: &KernelSchedule,
        ) -> Result<(), KernelCompilationError<std::io::Error>> {
            if target.0 {
                Ok(())
            } else {
                Err(KernelCompilationError::Unavailable { message: "fixture target unavailable".to_owned() })
            }
        }
        fn configuration_key(
            &self,
            target: &FixtureTarget,
            options: &u32,
            _schedule: &KernelSchedule,
        ) -> Result<Vec<u8>, KernelCompilationError<std::io::Error>> {
            let mut key = vec![u8::from(target.0)];
            key.extend_from_slice(&options.to_le_bytes());
            Ok(key)
        }
        fn compile(
            &self,
            _kernel: &VerifiedKernel<'_>,
            _target: &FixtureTarget,
            options: &u32,
            _schedule: &KernelSchedule,
        ) -> Result<(), KernelCompilationError<std::io::Error>> {
            if *options == u32::MAX {
                return Err(KernelCompilationError::Compiler(std::io::Error::other("fixture compiler failed")));
            }
            Ok(())
        }
    }

    /// Deferred hand-authored output whose identity includes its actual handler selection.
    struct Embedding(&'static str);

    impl KernelOutputEmbedding<()> for Embedding {
        fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
            Ok(self.0.as_bytes().to_vec())
        }
        fn custom_call(
            &self,
            kernel: &VerifiedKernel<'_>,
            _output: &(),
        ) -> Result<CustomCallOperation, KernelEmbeddingError> {
            Ok(CustomCallOperation::new(
                self.0,
                vec![kernel.definition().operation().parameters()[0].r#type().into_owned()],
            )
            .with_input_output_alias(0, 0)?)
        }
    }

    /// Creates an ordinary XLA graph with a real attached scalar kernel body.
    fn program() -> FlatXlaProgram {
        let definition = crate::kernels::tests::definition();
        TracingContext::<XlaConstant, XlaOperation>::trace(
            |inputs: Vec<Tracer<TracingContext<XlaConstant, XlaOperation>>>| {
                stage_kernel(inputs[0].context(), &definition, &inputs)
            },
            definition.operation().input_types(),
        )
        .unwrap()
        .1
    }

    #[test]
    fn test_stage_kernel_unused_outputs_preserve_assertions() {
        use ryft_core::{
            ArrayIrOperation, DimensionBounds, DimensionFromScalarOperation, DimensionVariable, ReferenceRead,
            ReferenceWrite,
        };
        let logical = crate::kernels::tests::definition().operation().clone();
        let definition = KernelDefinition::trace(logical, |(references, _)| {
            let value = references[0].read()?;
            references[0].context().bind(
                ArrayIrOperation::DimensionFromScalar(DimensionFromScalarOperation::new(DimensionVariable::new(
                    "checked",
                    DimensionBounds::new(0, Some(10)).unwrap(),
                ))),
                vec![],
                &[value.clone()],
            )?;
            references[0].write(&value)?;
            Ok(())
        })
        .unwrap();
        let (_, program) = TracingContext::<XlaConstant, XlaOperation>::trace(
            |inputs: Vec<Tracer<TracingContext<XlaConstant, XlaOperation>>>| {
                // The complete kernel result is deliberately unused by its enclosing computation.
                stage_kernel(inputs[0].context(), &definition, &inputs)?;
                Ok(Vec::<Tracer<TracingContext<XlaConstant, XlaOperation>>>::new())
            },
            definition.operation().input_types(),
        )
        .unwrap();
        assert_eq!(program.output_count(), 0);
        assert!(program.effects().classes().contains(EffectClass::OrderedAssertion));
        let simplified = program.simplified().unwrap();
        assert_eq!(simplified.output_count(), 0);
        assert_eq!(simplified.instructions().len(), 1);
        assert!(matches!(simplified.instructions()[0].operation(), XlaOperation::Kernel(_)));
        assert!(simplified.effects().classes().contains(EffectClass::OrderedAssertion));
        let nested_assertions = simplified
            .entry_region_ref()
            .instructions_in_closure()
            .filter(|(_, instruction)| instruction.operation().name() == "dimension_from_scalar")
            .count();
        assert_eq!(nested_assertions, 1);
    }

    #[test]
    fn test_xla_kernel_operation_new() {
        let definition = crate::kernels::tests::definition();
        let carrier = XlaKernelOperation::new(KernelOperation::Call(definition.operation().clone()));
        assert_eq!(carrier.name(), definition.operation().name());
        assert_eq!(carrier.region_slots(), definition.operation().region_slots());
        assert_eq!(carrier.input_region_provenance(0, 0), InputRegionProvenance::Local);
    }

    #[test]
    fn test_stage_body() {
        let definition = crate::kernels::tests::definition();
        let body = stage_body(&definition).unwrap();
        assert_eq!(body.input_types(), definition.body().input_types());
        assert_eq!(body.regions().len(), definition.body().regions().len());
        let restored = definition_from_body(definition.operation(), body.entry_region_ref()).unwrap();
        assert_eq!(restored.semantic_key().unwrap(), definition.semantic_key().unwrap());
        assert_eq!(restored.body().effects().classes(), definition.body().effects().classes());
    }

    #[test]
    fn test_stage_body_materializes_array_literals() {
        let operation = crate::kernels::tests::definition().operation().clone();
        let mut builder = ryft_core::ProgramBuilder::<ArrayIrValue<CpuArray>, KernelOperation>::new();
        let reference = builder.add_input(operation.body_input_types()[0].clone());
        let literal = builder.add_constant(ArrayIrValue::Array(CpuArray::scalar(42_i32).unwrap()));
        builder
            .add_instruction(
                ryft_core::ReferenceWriteOperation::<ryft_core::ArrayType, ArrayIrType>::new(),
                vec![],
                vec![reference, literal],
                None,
            )
            .unwrap();
        let definition =
            KernelDefinition::new(operation, builder.build(vec![], vec![Placeholder], vec![]).unwrap()).unwrap();
        let staged = stage_body(&definition).unwrap();
        assert_eq!(staged.instructions().len(), 2);
        assert!(
            matches!(staged.instructions()[0].operation(), XlaOperation::Array(ryft_core::ArrayOperation::Constant(value))
            if value.value() == &CpuArray::scalar(42_i32).unwrap())
        );
        assert_eq!(staged.instructions()[0].outputs(), &[literal]);
        let restored = definition_from_body(definition.operation(), staged.entry_region_ref()).unwrap();
        assert_eq!(
            restored.interpret(vec![CpuArray::scalar(0_i32).unwrap()], 1).unwrap(),
            vec![CpuArray::scalar(42_i32).unwrap()]
        );
    }

    #[test]
    fn test_stage_kernel() {
        let program = program();
        assert_eq!(program.regions().len(), 2);
        assert_eq!(program.instructions()[0].regions().len(), 1);
        assert!(program.effects().classes().contains(EffectClass::OrderedState));
    }

    #[test]
    fn test_xla_kernel_execution_facts_from_client() {
        let plugin = ryft_pjrt::load_cpu_plugin().unwrap();
        let client = plugin
            .client(ryft_pjrt::ClientOptions::CPU(ryft_pjrt::CpuClientOptions {
                device_count: Some(1),
                ..Default::default()
            }))
            .unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = ryft_core::DeviceMesh::new(
            ryft_core::LogicalMesh::new(vec![]).unwrap(),
            devices.iter().map(|device| ryft_core::Device::from_pjrt(device.clone()).unwrap()).collect(),
        )
        .unwrap();
        let facts = XlaKernelExecutionFacts::from_client(&client, &mesh).unwrap();
        assert_eq!(facts.platform_name, client.platform_name().unwrap());
        assert_eq!(facts.platform_version, client.platform_version().unwrap());
        assert_eq!(facts.pjrt_version, client.version());
        assert_eq!(facts.has_ffi_extension, client.ffi_extension().is_ok());
        assert_eq!(facts.devices.len(), 1);
        assert_eq!(facts.devices[0].kind, devices[0].kind().unwrap());
        assert_eq!(
            facts.devices[0].attributes,
            devices[0].attributes().unwrap().iter().map(|(name, value)| (name.clone(), value.clone())).collect()
        );
    }

    #[test]
    fn test_xla_kernel_execution_facts_configuration_key() {
        let first = facts().unwrap();
        let key = first.configuration_key().unwrap();
        assert_eq!(key, first.clone().configuration_key().unwrap());
        let mut changed = first.clone();
        changed.platform_name = "other".to_owned();
        assert_ne!(key, changed.configuration_key().unwrap());
        changed = first.clone();
        changed.platform_version = "2".to_owned();
        assert_ne!(key, changed.configuration_key().unwrap());
        changed = first.clone();
        changed.pjrt_version.minor += 1;
        assert_ne!(key, changed.configuration_key().unwrap());
        changed = first.clone();
        changed.has_ffi_extension = false;
        assert_ne!(key, changed.configuration_key().unwrap());
        changed = first.clone();
        changed
            .devices
            .push(XlaKernelDeviceFacts { kind: "device".to_owned(), attributes: Default::default() });
        assert_ne!(key, changed.configuration_key().unwrap());
        let before_attribute = changed.configuration_key().unwrap();
        changed.devices[0]
            .attributes
            .insert("compute_capability".to_owned(), ryft_pjrt::Value::String("12.1".to_owned()));
        assert_ne!(before_attribute, changed.configuration_key().unwrap());
        for (first_value, second_value) in [
            (ryft_pjrt::Value::Bool(true), ryft_pjrt::Value::String("true".to_owned())),
            (ryft_pjrt::Value::I64(1), ryft_pjrt::Value::I64List(vec![1])),
            (ryft_pjrt::Value::F32(f32::from_bits(0x7fc0_0001)), ryft_pjrt::Value::F32(f32::from_bits(0x7fc0_0002))),
        ] {
            changed.attributes.insert("value".to_owned(), first_value);
            let first_key = changed.configuration_key().unwrap();
            changed.attributes.insert("value".to_owned(), second_value);
            assert_ne!(first_key, changed.configuration_key().unwrap());
        }
    }

    #[test]
    fn test_xla_kernel_compiler_binding_new() {
        let first = XlaKernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            Embedding("first"),
            1,
        )
        .unwrap();
        let same = XlaKernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            Embedding("first"),
            1,
        )
        .unwrap();
        let different = XlaKernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            Embedding("second"),
            1,
        )
        .unwrap();
        assert_eq!(first, same);
        assert_ne!(first, different);
    }

    #[test]
    fn test_xla_kernel_compiler_binding_compile() {
        let binding = XlaKernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            Embedding("fixture"),
            1,
        )
        .unwrap();
        let definition = crate::kernels::tests::definition();
        let first_facts = facts().unwrap();
        let first = binding.compile(&definition, &first_facts).unwrap();
        let mut changed = first_facts.clone();
        changed.platform_version = "2".to_owned();
        let second = binding.compile(&definition, &changed).unwrap();
        assert_ne!(first.custom_call().attributes(), second.custom_call().attributes());
        changed.platform_name = "other".to_owned();
        assert!(matches!(binding.compile(&definition, &changed), Err(KernelEmbeddingError::Invalid { message })
            if message == "fixture execution platform mismatch"));
        // Target mismatch wins over adapter rejection, proving the execution check precedes compiler admission.
        let unavailable = XlaKernelCompilerBinding::new(
            Compiler,
            FixtureTarget(false),
            1,
            KernelSchedule::default(),
            Embedding("fixture"),
            1,
        )
        .unwrap();
        assert!(matches!(unavailable.compile(&definition, &changed), Err(KernelEmbeddingError::Invalid { message })
            if message == "fixture execution platform mismatch"));
    }

    #[test]
    fn test_xla_kernel_compiler_binding_compile_rejects_changed_embedding() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// Embedding with externally mutable configuration to exercise the immutable binding contract.
        struct MutableEmbedding {
            /// Configuration shared with the caller after the binding is created.
            configuration: Arc<AtomicUsize>,
        }

        impl KernelOutputEmbedding<()> for MutableEmbedding {
            fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
                Ok(self.configuration.load(Ordering::SeqCst).to_le_bytes().to_vec())
            }

            fn custom_call(
                &self,
                kernel: &VerifiedKernel<'_>,
                output: &(),
            ) -> Result<CustomCallOperation, KernelEmbeddingError> {
                Embedding("fixture").custom_call(kernel, output)
            }
        }

        let configuration = Arc::new(AtomicUsize::new(0));
        let binding = XlaKernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            MutableEmbedding { configuration: Arc::clone(&configuration) },
            1,
        )
        .unwrap();
        let definition = crate::kernels::tests::definition();
        assert!(binding.compile(&definition, &facts().unwrap()).is_ok());
        configuration.store(1, Ordering::SeqCst);
        assert!(matches!(
            binding.compile(&definition, &facts().unwrap()),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "selected kernel embedding configuration changed after binding",
        ));
    }

    #[test]
    fn test_select_kernels() {
        let program = program();
        assert!(program.effects().classes().contains(EffectClass::OrderedState));
        let binding = XlaKernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            Embedding("ryft.test.selected"),
            1,
        )
        .unwrap();
        let selected = select_kernels(&program, Some(&binding), facts).unwrap().unwrap();
        assert!(selected.effects().classes().is_empty());
        assert_eq!(selected.regions().len(), 1);
        assert_eq!(selected.input_types(), program.input_types());
        assert_eq!(selected.output_types(), program.output_types());
        assert!(matches!(selected.instructions()[0].operation(), XlaOperation::CustomCall(_)));
    }

    #[test]
    fn test_select_kernels_missing_binding() {
        assert!(matches!(select_kernels(&program(), None, facts), Err(KernelEmbeddingError::Invalid { message })
            if message == "kernel call has no explicitly enabled XLA compiler binding"));
    }

    #[test]
    fn test_select_kernels_admission_failure() {
        let binding = XlaKernelCompilerBinding::new(
            Compiler,
            FixtureTarget(false),
            1,
            KernelSchedule::default(),
            Embedding("ryft.test.selected"),
            1,
        )
        .unwrap();
        assert!(matches!(select_kernels(&program(), Some(&binding), facts), Err(KernelEmbeddingError::Compiler(error))
            if error.to_string() == "kernel compiler is unavailable: fixture target unavailable"));
    }

    #[test]
    fn test_select_kernels_compiler_failure_does_not_poison_later_selection() {
        let program = program();
        let original = program.to_string();
        let failing = XlaKernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            u32::MAX,
            KernelSchedule::default(),
            Embedding("fixture"),
            1,
        )
        .unwrap();
        let successful = XlaKernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            Embedding("fixture"),
            1,
        )
        .unwrap();
        let failure = select_kernels(&program, Some(&failing), facts).unwrap_err();
        assert!(
            matches!(&failure, KernelEmbeddingError::Compiler(error) if error.to_string() == "fixture compiler failed")
        );
        assert_eq!(std::error::Error::source(&failure).unwrap().to_string(), "fixture compiler failed");
        assert_eq!(program.to_string(), original);
        let first = select_kernels(&program, Some(&successful), facts).unwrap().unwrap();
        let second = select_kernels(&program, Some(&successful), facts).unwrap().unwrap();
        assert_eq!(first.to_string(), second.to_string());
        assert!(matches!(first.instructions()[0].operation(), XlaOperation::CustomCall(_)));
        assert!(first.effects().classes().is_empty());
        assert_eq!(first.output_types(), program.output_types());
    }

    #[test]
    fn test_select_kernels_revalidates_current_body_effects() {
        let program = program();
        let regions = program
            .regions()
            .iter()
            .enumerate()
            .map(|(index, region)| {
                let mut instructions = region.instructions().to_vec();
                if index == 0 {
                    instructions.push(Instruction::new(
                        XlaOperation::Kernel(XlaKernelOperation::new(KernelOperation::Portable(
                            ryft_core::ArrayOperation::<CpuArray>::CustomCall(
                                CustomCallOperation::new("test.external", vec![])
                                    .with_effect_class(EffectClass::OrderedIo),
                            )
                            .into(),
                        ))),
                        vec![],
                        vec![],
                        vec![],
                    ));
                }
                Region::new(
                    region.atoms().to_vec(),
                    region.input_ids().to_vec(),
                    region.output_ids().to_vec(),
                    instructions,
                )
            })
            .collect();
        let changed = Program::new(vec![Placeholder], vec![Placeholder], regions, program.entry()).unwrap();
        let binding = XlaKernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            Embedding("ryft.test.selected"),
            1,
        )
        .unwrap();
        assert!(matches!(
            select_kernels(&changed, Some(&binding), facts),
            Err(KernelEmbeddingError::UnsupportedEffect { effect: EffectClass::OrderedIo })
        ));
    }
}
