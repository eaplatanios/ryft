//! XLA staging carriers for portable and enabled adapter kernels with ordinary attached computation regions.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use ryft_core::kernels::{
    KernelCallOperation, KernelCompiler, KernelDefinition, KernelExtension, KernelExtensionMemory, KernelOperation,
    KernelSchedule, NoKernelExtension, VerifiedKernel,
};
use ryft_core::operations::custom_call::CustomCallOperation;
use ryft_core::{
    Array as CpuArray, ArrayIrType, ArrayIrValue, ArrayReferenceView, Atom, AtomId, ConstantOperation, Context,
    CotangentAccumulator, DifferentiableOperation, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy, Domain, Effects, InputRegionProvenance, Instruction,
    InterpretableOperation, InterpretationDriver, MaybeZero, Operation, OutputRegionProvenance, PartialValue,
    PartiallyEvaluatableOperation, Placeholder, Program, ProgramError, ReferenceAccessMode, ReferenceDischargeContext,
    ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue, ReferenceDischargeableOperation,
    ReferenceViewOperation, ReferenceViewValidationError, Region, RegionInterface, RegionSlot, Tracer, TracingContext,
    TransposableOperation, TranspositionContext, TranspositionDriver, Type, TypeError, TypeIdentityRenaming, Typed,
    Value,
};

use crate::experimental::ops::{FlatXlaProgram, XlaConstant, XlaOperation};
use crate::kernels::{CompiledKernel, KernelEmbeddingError, KernelOutputEmbedding};

/// Adapter operation families enabled in this XLA integration.
///
/// Portable operations remain ordinary `KernelOperation` variants. This enum owns only the typed conversion between
/// enabled adapter families and XLA staging; it does not choose a compiler or erase extension semantics.
#[derive(Clone, Debug)]
pub enum XlaKernelExtension {
    /// Exact Mosaic GPU instruction semantics, admitted only by an explicitly selected Mosaic compiler.
    #[cfg(feature = "mosaic-gpu")]
    Mosaic(ryft_mosaic::kernels::gpu::GpuOperation),
}

impl From<NoKernelExtension> for XlaKernelExtension {
    fn from(extension: NoKernelExtension) -> Self {
        match extension {}
    }
}

impl TryFrom<XlaKernelExtension> for NoKernelExtension {
    type Error = ProgramError;

    fn try_from(extension: XlaKernelExtension) -> Result<Self, Self::Error> {
        match extension {
            #[cfg(feature = "mosaic-gpu")]
            XlaKernelExtension::Mosaic(_) => Err(ProgramError::UnsupportedOperation {
                message: "portable kernel compiler binding cannot consume a Mosaic GPU extension".to_owned(),
            }),
        }
    }
}

#[cfg(feature = "mosaic-gpu")]
impl From<ryft_mosaic::kernels::gpu::GpuOperation> for XlaKernelExtension {
    fn from(extension: ryft_mosaic::kernels::gpu::GpuOperation) -> Self {
        Self::Mosaic(extension)
    }
}

#[cfg(feature = "mosaic-gpu")]
impl TryFrom<XlaKernelExtension> for ryft_mosaic::kernels::gpu::GpuOperation {
    type Error = ProgramError;

    fn try_from(extension: XlaKernelExtension) -> Result<Self, Self::Error> {
        match extension {
            XlaKernelExtension::Mosaic(extension) => Ok(extension),
        }
    }
}

impl Display for XlaKernelExtension {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl ReferenceViewOperation for XlaKernelExtension {
    type View = ArrayReferenceView;

    fn reference_view(&self, _output_index: usize) -> Option<ArrayReferenceView> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.reference_view(_output_index),
        }
    }

    fn validate_reference_view(
        view: &ArrayReferenceView,
        source: &ArrayIrType,
        target: &ArrayIrType,
    ) -> Result<(), ReferenceViewValidationError> {
        ryft_core::validate_array_reference_view(view, source, target)
    }

    fn reapply_reference_view<C: Context<Type = ArrayIrType, Operation = Self>>(
        _context: &C,
        _view: &ArrayReferenceView,
        _source: C::Value,
        _symbols: &[C::Value],
    ) -> Result<C::Value, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: "XLA extension family cannot construct portable reference views".to_owned(),
        })
    }
}

impl KernelExtension for XlaKernelExtension {
    fn semantic_key(&self) -> Result<Vec<u8>, TypeError> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.semantic_key(),
        }
    }

    fn memory_semantics(&self) -> Result<KernelExtensionMemory, TypeError> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.memory_semantics(),
        }
    }
}

impl Operation for XlaKernelExtension {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.name(),
        }
    }

    fn region_slots(&self) -> &'static [RegionSlot] {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.region_slots(),
        }
    }

    fn infer_region_input_types(
        &self,
        _input_types: &[ArrayIrType],
        _region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.infer_region_input_types(_input_types, _region_interfaces),
        }
    }

    fn infer_output_types(
        &self,
        _input_types: &[ArrayIrType],
        _region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.infer_output_types(_input_types, _region_interfaces),
        }
    }

    fn input_region_provenance(&self, _region_index: usize, _input_index: usize) -> InputRegionProvenance {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.input_region_provenance(_region_index, _input_index),
        }
    }

    fn output_region_provenance(&self, _output_index: usize) -> Vec<OutputRegionProvenance> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.output_region_provenance(_output_index),
        }
    }

    fn is_zero(&self, _output_index: usize) -> bool {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.is_zero(_output_index),
        }
    }

    fn region_capture_input_count(&self, _region_index: usize) -> Option<usize> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.region_capture_input_count(_region_index),
        }
    }

    fn reference_output_identity_input(&self, _output_index: usize) -> Option<usize> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.reference_output_identity_input(_output_index),
        }
    }

    fn allows_reference_access_through_region_input(&self, _region_index: usize, _mode: ReferenceAccessMode) -> bool {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.allows_reference_access_through_region_input(_region_index, _mode),
        }
    }

    fn effects(&self) -> Cow<'_, Effects> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.effects(),
        }
    }

    fn render(&self, formatter: &mut Formatter<'_>, _indentation: usize) -> std::fmt::Result {
        #[cfg(not(feature = "mosaic-gpu"))]
        let _ = &formatter;
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.render(formatter, _indentation),
        }
    }

    fn rename_type_identities(
        &self,
        _renaming: &TypeIdentityRenaming<<ArrayIrType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => Ok(Self::Mosaic(operation.rename_type_identities(_renaming)?)),
        }
    }
}

/// Canonical kernel instruction retained inside the XLA operation family until explicit compiler selection.
///
/// The payload contains only operation metadata. Its computations remain ordinary attached regions of the XLA
/// program, so effect, reference, identity, and structural validation continue to inspect the actual current body.
/// Generic differentiation and reference discharge require an owner-supported kernel rule and reject this carrier.
#[derive(Clone, Debug)]
pub struct XlaKernelOperation(pub(crate) KernelOperation<XlaKernelExtension>);

impl XlaKernelOperation {
    /// Wraps a canonical portable instruction without changing its semantic contract.
    pub fn new(operation: KernelOperation) -> Self {
        Self::from(operation)
    }

    /// Returns the canonical operation metadata; computations belong to the containing instruction's regions.
    pub fn operation(&self) -> &KernelOperation<XlaKernelExtension> {
        &self.0
    }
}

impl<Extension: Operation<Type = ArrayIrType> + Into<XlaKernelExtension>> From<KernelOperation<Extension>>
    for XlaKernelOperation
{
    fn from(operation: KernelOperation<Extension>) -> Self {
        // This conversion only widens a supported family and cannot fail.
        Self(
            operation
                .map_extension(|extension| match Into::<XlaKernelExtension>::into(extension) {
                    #[cfg(feature = "mosaic-gpu")]
                    extension @ XlaKernelExtension::Mosaic(_) => Ok(extension),
                })
                .unwrap(),
        )
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
fn stage_body<Extension: KernelExtension + Into<XlaKernelExtension>>(
    definition: &KernelDefinition<Extension>,
) -> Result<FlatXlaProgram, ProgramError> {
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
                    XlaOperation::Kernel(XlaKernelOperation::from(instruction.operation().clone())),
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

/// Binds an unselected kernel with its real body attached to the surrounding XLA program.
///
/// Portable instructions and enabled typed extensions remain visible to ordinary region, effect, and reference
/// traversal. Compiler selection belongs to the enclosing XLA domain's immutable compilation options. Staging itself
/// neither invokes an adapter nor acquires native runtime resources.
pub fn stage_kernel<C, Extension>(
    context: &C,
    definition: &KernelDefinition<Extension>,
    inputs: &[C::Value],
) -> Result<Vec<C::Value>, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = XlaConstant, Operation = XlaOperation>,
    Extension: KernelExtension + Into<XlaKernelExtension>,
{
    context.bind(
        XlaOperation::Kernel(XlaKernelOperation::new(KernelOperation::<NoKernelExtension>::Call(
            definition.operation().clone(),
        ))),
        vec![stage_body(definition)?],
        inputs,
    )
}

/// Restores the current attached body for the core verifier; retained operation metadata is never a body proof.
fn definition_from_body(
    operation: &KernelCallOperation,
    body: ryft_core::RegionRef<'_, XlaConstant, XlaOperation>,
) -> Result<KernelDefinition<XlaKernelExtension>, ProgramError> {
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
                                "XLA operation `{}` has no kernel reconstruction",
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
        dyn Fn(
                &KernelDefinition<XlaKernelExtension>,
                &XlaKernelExecutionFacts,
            ) -> Result<CustomCallOperation, KernelEmbeddingError>
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
        Self::new_with_extensions::<C, B, NoKernelExtension>(
            compiler,
            target,
            options,
            schedule,
            embedding,
            maximum_programs,
        )
    }

    /// Binds an explicitly selected extension family without changing portable constructor inference.
    ///
    /// Before compilation, the current attached body is reconstructed into this exact family. An operation owned by
    /// another family fails conversion rather than falling back to a different compiler. Validation and embedding
    /// then use the same execution-facts and immutable-configuration path as ordinary portable kernels.
    pub fn new_with_extensions<C, B, Extension>(
        compiler: C,
        target: C::Target,
        options: C::Options,
        schedule: KernelSchedule,
        embedding: B,
        maximum_programs: usize,
    ) -> Result<Self, KernelEmbeddingError>
    where
        C: 'static + Send + Sync + KernelCompiler<Extension, Error: 'static + Send + Sync>,
        C::Target: 'static + Send + Sync + XlaKernelTarget,
        C::Options: 'static + Send + Sync,
        B: 'static + Send + Sync + KernelOutputEmbedding<C::Output, Extension>,
        Extension: 'static + Send + Sync + KernelExtension + TryFrom<XlaKernelExtension, Error = ProgramError>,
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
        let compile =
            Arc::new(move |definition: &KernelDefinition<XlaKernelExtension>, facts: &XlaKernelExecutionFacts| {
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
                let body = definition
                    .body()
                    .map_operations(|operation| operation.clone().map_extension(Extension::try_from))
                    .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
                let definition = KernelDefinition::new(definition.operation().clone(), body)
                    .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
                let verified = VerifiedKernel::new(&definition, maximum_programs)
                    .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
                let output = verified
                    .compile(&compiler, &target, &options, &schedule)
                    .map_err(|error| KernelEmbeddingError::Compiler(Box::new(error)))?;
                let facts_configuration = facts.configuration_key()?;
                // The common embedding boundary appends the embedding key exactly once. This component retains
                // compiler and execution-facts identity; the binding separately checks both captured configurations.
                let mut configuration = Vec::new();
                configuration.extend_from_slice(&(adapter_configuration.len() as u64).to_le_bytes());
                configuration.extend_from_slice(&adapter_configuration);
                configuration.extend_from_slice(&(facts_configuration.len() as u64).to_le_bytes());
                configuration.extend_from_slice(&facts_configuration);
                Ok(CompiledKernel::from_output(&verified, &configuration, &output, &embedding)?.custom_call().clone())
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
        definition: &KernelDefinition<XlaKernelExtension>,
        facts: &XlaKernelExecutionFacts,
    ) -> Result<CustomCallOperation, KernelEmbeddingError> {
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
                                compiled.into(),
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
        let definition = KernelDefinition::<NoKernelExtension>::trace(logical, |(references, _)| {
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
        let portable = KernelDefinition::new(
            restored.operation().clone(),
            restored
                .body()
                .map_operations(|operation| operation.clone().map_extension(NoKernelExtension::try_from))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            portable.interpret(vec![CpuArray::scalar(0_i32).unwrap()], 1).unwrap(),
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

    #[cfg(feature = "mosaic-gpu")]
    #[test]
    fn test_xla_kernel_compiler_binding_new_with_extensions() {
        use ryft_core::kernels::{Grid, KernelParameterAccess, whole_array_parameter};
        use ryft_core::{ArrayType, DataType, ReferenceRead, ReferenceWrite};
        use ryft_mosaic::kernels::gpu::{Compiler as GpuCompiler, GpuOperation, Mma, Options, Target};

        use crate::kernels::MosaicGpuEmbedding;

        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(
                    ArrayType::new_static(DataType::F16, vec![64, 16]),
                    KernelParameterAccess::ReadOnly,
                )
                .unwrap(),
                whole_array_parameter(
                    ArrayType::new_static(DataType::F16, vec![16, 8]),
                    KernelParameterAccess::ReadOnly,
                )
                .unwrap(),
                whole_array_parameter(
                    ArrayType::new_static(DataType::F32, vec![64, 8]),
                    KernelParameterAccess::WriteOnly,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let definition = KernelDefinition::<GpuOperation>::trace(call, |(references, _)| {
            let left = references[0].read()?;
            let right = references[1].read()?;
            let output = references[0].context().bind(
                KernelOperation::Extension(GpuOperation::Wgmma),
                vec![],
                &[left, right],
            )?;
            references[2].write(&output[0])
        })
        .unwrap();
        let (_, program) = TracingContext::<XlaConstant, XlaOperation>::trace(
            |inputs: Vec<Tracer<TracingContext<XlaConstant, XlaOperation>>>| {
                stage_kernel(inputs[0].context(), &definition, &inputs)
            },
            definition.operation().input_types(),
        )
        .unwrap();
        let body_id = program.instructions()[0].regions()[0];
        assert_eq!(
            program
                .region_ref(body_id)
                .unwrap()
                .instructions()
                .iter()
                .filter(|instruction| {
                    matches!(
                        instruction.operation(),
                        XlaOperation::Kernel(XlaKernelOperation(KernelOperation::Extension(
                            XlaKernelExtension::Mosaic(GpuOperation::Wgmma)
                        )))
                    )
                })
                .count(),
            1
        );
        let restored = definition_from_body(definition.operation(), program.region_ref(body_id).unwrap()).unwrap();
        assert_eq!(restored.semantic_key().unwrap(), definition.semantic_key().unwrap());

        let target = Target::new(9, 0).unwrap().with_threads_per_block(128).unwrap();
        let options = Options::default().with_mma(Mma::Wgmma);
        let gpu_facts = || {
            Ok(XlaKernelExecutionFacts {
                platform_name: "CUDA".to_owned(),
                platform_version: "cuda 13020".to_owned(),
                pjrt_version: ryft_pjrt::VERSION,
                has_ffi_extension: true,
                attributes: BTreeMap::new(),
                devices: vec![XlaKernelDeviceFacts {
                    kind: "fixture GPU".to_owned(),
                    attributes: BTreeMap::from([(
                        "compute_capability".to_owned(),
                        ryft_pjrt::Value::String("9.0".to_owned()),
                    )]),
                }],
            })
        };
        let portable = XlaKernelCompilerBinding::new(
            GpuCompiler,
            target.clone(),
            options.clone(),
            KernelSchedule::default(),
            MosaicGpuEmbedding,
            1,
        )
        .unwrap();
        assert!(matches!(select_kernels(&program, Some(&portable), gpu_facts),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "portable kernel compiler binding cannot consume a Mosaic GPU extension"));

        let binding = XlaKernelCompilerBinding::new_with_extensions::<_, _, GpuOperation>(
            GpuCompiler,
            target,
            options,
            KernelSchedule::default(),
            MosaicGpuEmbedding,
            1,
        )
        .unwrap();
        // Identical compiler configuration never bypasses the current body's family admission.
        assert_eq!(portable.configuration(), binding.configuration());
        let selected = select_kernels(&program, Some(&binding), gpu_facts).unwrap().unwrap();
        assert_eq!(selected.instructions().len(), 1);
        assert_eq!(selected.instructions()[0].operation().name(), "custom_call");
        assert_eq!(selected.output_types(), definition.operation().output_types());
        assert!(!selected.effects().classes().contains(EffectClass::OrderedState));
        assert_eq!(selected.regions().len(), 1);
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
        let portable = crate::kernels::tests::definition();
        let staged = stage_body(&portable).unwrap();
        let definition = definition_from_body(portable.operation(), staged.entry_region_ref()).unwrap();
        let first_facts = facts().unwrap();
        let first = binding.compile(&definition, &first_facts).unwrap();
        let mut changed = first_facts.clone();
        changed.platform_version = "2".to_owned();
        let second = binding.compile(&definition, &changed).unwrap();
        assert_ne!(first.attributes(), second.attributes());
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
        let portable = crate::kernels::tests::definition();
        let staged = stage_body(&portable).unwrap();
        let definition = definition_from_body(portable.operation(), staged.entry_region_ref()).unwrap();
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
