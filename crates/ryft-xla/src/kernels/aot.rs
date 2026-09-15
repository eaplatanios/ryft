//! Complete kernel deployments over the existing XLA executable persistence contract.
//!
//! Bundles carry checked portable source, lowering diagnostics, compatibility metadata, and the complete serialized
//! PJRT executable, including backend artifacts. Import never recompiles. Source eligibility is the core codec's
//! contract; callers can explicitly recompile the retained source when compatibility changes. Checksums detect
//! corruption, not an untrusted producer: executable bundles must come from a trusted source.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ryft_core::kernels::{KernelDefinition, VerifiedKernel};
use ryft_core::{
    AnalyzableCompilationDomain, ArrayIrType, ArrayType, CompilationCacheDomain, CompilationDomain,
    CompilationStagingRequest, CompilationTracer, DeviceMesh,
};
use ryft_pjrt::Execution;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::experimental::XlaDomainError;
use crate::experimental::domains::XlaCompiledProgram;
use crate::kernels::KernelEmbeddingError;
use crate::kernels::staging::{XlaKernelCompilerBinding, XlaKernelExecutionFacts, stage_kernel};
use crate::{Array, XlaCompilationAnalysis, XlaDomain, XlaOptions};

/// A bundle is malformed, incompatible, or cannot be produced by the selected backend.
#[derive(Debug, Error)]
pub enum KernelAotError {
    /// A bounded envelope or compatibility check failed before native loading.
    #[error("invalid kernel AOT bundle: {message}")]
    Invalid {
        /// Exact failed invariant.
        message: String,
    },

    /// Source serialization or metadata decoding failed.
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),

    /// Source verification or compiler embedding failed.
    #[error(transparent)]
    Embedding(#[from] KernelEmbeddingError),

    /// Existing XLA compilation, persistence, or runtime checks failed.
    #[error(transparent)]
    Xla(#[from] XlaDomainError),
}

/// Timings and compiler diagnostics correlated with one exact kernel definition.
///
/// Lowering includes adapter compilation; compilation includes the ordinary executable cache lookup. These are host
/// wall times, not device launch durations. PJRT's existing analysis retains optional counters and raw properties;
/// unsupported register, spill, occupancy, and TMEM measurements must not be inferred from other memory counters.
#[derive(Clone, Debug, Serialize)]
pub struct KernelCompilationReport {
    /// SHA-256 of the canonical semantic identity.
    pub semantic_digest: String,
    /// SHA-256 of the selected compiler/target/options/embedding identity.
    pub configuration_digest: String,
    /// Instruction provenance descriptions, in source order.
    pub provenance: Vec<String>,
    /// Host time in source staging and lowering, including adapter selection.
    pub lowering_duration: Duration,
    /// Host time in the XLA compilation or executable cache lookup.
    pub compilation_duration: Duration,
    /// Existing typed PJRT analysis, preserving unsupported properties as absent.
    pub analysis: XlaCompilationAnalysis,
}

/// Private, versioned bundle metadata. Binary source/executable sections avoid JSON byte-array expansion.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    /// Exact compiler binding configuration, without querying its compiler installation.
    configuration: Vec<u8>,
    /// Exact live platform, plugin, and selected-device facts.
    execution: Vec<u8>,
    /// Canonical semantic digest, checked after decoding the source.
    semantic_digest: String,
    /// StableHLO emitted by the ordinary lowering, including selected artifacts and source locations.
    stable_hlo: String,
    /// Existing report's JSON representation, retained for offline inspection without a second analysis schema.
    report: serde_json::Value,
}

/// Relocatable, validated source and executable bundle. No path or runtime pointer is an artifact identifier.
#[derive(Clone, Debug)]
pub struct KernelAotBundle {
    /// Decoded canonical source; callers cannot mutate it behind its executable.
    definition: KernelDefinition,
    /// Exact already-validated transport bytes for source and diagnostic provenance.
    source: Vec<u8>,
    /// Existing XLA persistence envelope, including native backend resources.
    executable: Vec<u8>,
    /// Compatibility and inspection data protected by the outer checksum.
    manifest: Manifest,
}

impl KernelAotBundle {
    /// Bundle schema independent of the core source and XLA executable schemas, each validated by its owner.
    const MAGIC: &[u8; 8] = b"RYFTKA01";
    /// Maximum accepted/exported total bundle size, including source, metadata, and executable.
    const MAXIMUM_BYTES: usize = 256 * 1024 * 1024;
    /// Maximum encoded source and metadata section sizes, checked before their JSON decoders run.
    const MAXIMUM_JSON_BYTES: usize = 64 * 1024 * 1024;
    /// Magic, three section lengths, and SHA-256 over all section lengths and contents.
    const HEADER_BYTES: usize = 8 + 3 * 8 + 32;

    /// Compiles a capture-free portable kernel using ordinary XLA staging, caching, analysis, and persistence.
    ///
    /// Source serialization and core verification precede compiler invocation. `options` must select a kernel
    /// compiler and the kernel must have at least one ordinary input. Unsupported source codecs or PJRT serialization
    /// are explicit errors; no fallback is selected.
    /// `maximum_programs` bounds source verification, not native compilation; adapter compiler deadlines remain active.
    pub fn compile<'c>(
        definition: &KernelDefinition,
        domain: &XlaDomain<'c>,
        options: XlaOptions,
        maximum_programs: usize,
    ) -> Result<Self, KernelAotError> {
        VerifiedKernel::new(definition, maximum_programs).map_err(|error| Self::invalid(error.to_string()))?;
        if definition.operation().input_types().is_empty() {
            return Err(Self::invalid("aot compilation requires at least one ordinary kernel input"));
        }
        let source = serde_json::to_vec(definition)?;
        if source.len() > Self::MAXIMUM_JSON_BYTES {
            return Err(Self::invalid("source exceeds the bundle size limit"));
        }
        let binding =
            options.kernel_compiler.as_ref().ok_or_else(|| Self::invalid("a compiler binding is required"))?;
        let configuration = binding.configuration().to_vec();
        let facts =
            XlaKernelExecutionFacts::from_client(domain.client().map_err(XlaDomainError::from)?, &options.mesh)?;
        let execution = facts.configuration_key()?;
        let start = Instant::now();
        let staged =
            domain.stage(CompilationStagingRequest::<XlaDomain<'c>, _, Vec<ArrayIrType>, Vec<ArrayIrType>>::new(
                |_, _, inputs: Vec<CompilationTracer<XlaDomain<'c>>>| {
                    Ok(stage_kernel(inputs[0].context(), definition, &inputs)?)
                },
                vec![],
                definition.operation().input_types(),
                options,
            ))?;
        let lowered = domain.lower(staged)?;
        let stable_hlo = lowered.lowered_program().stable_hlo().to_owned();
        let lowering_duration = start.elapsed();
        let start = Instant::now();
        let compiled = domain.compile(lowered)?;
        let compilation_duration = start.elapsed();
        let executable = domain
            .serialize_program(compiled.compiled_program())?
            .ok_or_else(|| Self::invalid("the plugin does not support executable serialization"))?;
        let semantic_digest = Self::semantic_digest(definition)?;
        let report = KernelCompilationReport {
            semantic_digest: semantic_digest.clone(),
            configuration_digest: format!("{:x}", Sha256::digest(&configuration)),
            provenance: definition
                .body()
                .regions()
                .iter()
                .flat_map(|region| region.instructions())
                .filter(|instruction| !instruction.provenance().is_unknown())
                .map(|instruction| instruction.provenance().to_string())
                .filter(|origin| !origin.is_empty())
                .collect(),
            lowering_duration,
            compilation_duration,
            analysis: domain.analyze(compiled.executable_function())?,
        };
        let bundle = Self {
            definition: definition.clone(),
            source,
            executable,
            manifest: Manifest {
                configuration,
                execution,
                semantic_digest,
                stable_hlo,
                report: serde_json::to_value(report)?,
            },
        };
        bundle.to_bytes()?;
        Ok(bundle)
    }

    /// Decodes a size-bounded bundle and revalidates its checksum, canonical source, and executable core proof.
    /// Native executable loading occurs only in [`Self::load`], after checking live compatibility.
    pub fn from_bytes(bytes: &[u8], maximum_programs: usize) -> Result<Self, KernelAotError> {
        if bytes.len() < Self::HEADER_BYTES || bytes.len() > Self::MAXIMUM_BYTES || &bytes[..8] != Self::MAGIC {
            return Err(Self::invalid("unsupported schema or bundle size"));
        }
        let expected: [u8; 32] = bytes[32..64].try_into().unwrap();
        let mut hash = Sha256::new();
        hash.update(&bytes[8..32]);
        hash.update(&bytes[64..]);
        if hash.finalize().as_slice() != expected {
            return Err(Self::invalid("bundle checksum mismatch"));
        }
        let mut offset = Self::HEADER_BYTES;
        let mut sections = Vec::with_capacity(3);
        for start in [8, 16, 24] {
            let length = usize::try_from(u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap()))
                .map_err(|_| Self::invalid("section length is not addressable"))?;
            let end = offset
                .checked_add(length)
                .filter(|end| *end <= bytes.len())
                .ok_or_else(|| Self::invalid("section exceeds the bundle boundary"))?;
            sections.push(&bytes[offset..end]);
            offset = end;
        }
        if offset != bytes.len() || sections.iter().any(|section| section.is_empty()) {
            return Err(Self::invalid("empty section or trailing bundle bytes"));
        }
        if sections[0].len() > Self::MAXIMUM_JSON_BYTES || sections[1].len() > Self::MAXIMUM_JSON_BYTES {
            return Err(Self::invalid("JSON section exceeds the size limit"));
        }
        let manifest: Manifest = serde_json::from_slice(sections[0])?;
        let definition: KernelDefinition = serde_json::from_slice(sections[1])?;
        VerifiedKernel::new(&definition, maximum_programs).map_err(|error| Self::invalid(error.to_string()))?;
        if manifest.semantic_digest != Self::semantic_digest(&definition)? {
            return Err(Self::invalid("source semantic identity mismatch"));
        }
        Ok(Self { definition, source: sections[1].to_vec(), executable: sections[2].to_vec(), manifest })
    }

    /// Returns the checked source for inspection or explicitly requested recompilation.
    pub fn definition(&self) -> &KernelDefinition {
        &self.definition
    }

    /// Returns selected StableHLO, including backend artifact attributes and source provenance.
    pub fn stable_hlo(&self) -> &str {
        &self.manifest.stable_hlo
    }

    /// Returns the original typed compilation report encoded as JSON for offline inspection.
    pub fn report(&self) -> &serde_json::Value {
        &self.manifest.report
    }

    /// Encodes the complete bundle with a versioned checksum; no external artifact paths are required.
    pub fn to_bytes(&self) -> Result<Vec<u8>, KernelAotError> {
        let manifest = serde_json::to_vec(&self.manifest)?;
        if manifest.len() > Self::MAXIMUM_JSON_BYTES || self.source.len() > Self::MAXIMUM_JSON_BYTES {
            return Err(Self::invalid("JSON section exceeds the size limit"));
        }
        let sections = [manifest.as_slice(), self.source.as_slice(), self.executable.as_slice()];
        let size = sections
            .iter()
            .try_fold(Self::HEADER_BYTES, |size, section| size.checked_add(section.len()))
            .filter(|size| *size <= Self::MAXIMUM_BYTES)
            .ok_or_else(|| Self::invalid("bundle exceeds the size limit"))?;
        let mut bytes = Vec::with_capacity(size);
        bytes.extend_from_slice(Self::MAGIC);
        for section in &sections {
            bytes.extend_from_slice(&(section.len() as u64).to_le_bytes());
        }
        bytes.resize(Self::HEADER_BYTES, 0);
        for section in sections {
            bytes.extend_from_slice(section);
        }
        let mut hash = Sha256::new();
        hash.update(&bytes[8..32]);
        hash.update(&bytes[64..]);
        bytes[32..64].copy_from_slice(&hash.finalize());
        Ok(bytes)
    }

    /// Loads through the existing XLA persistence decoder after exact compiler and live device checks.
    ///
    /// The binding is used only for identity; its compiler is never called. The mesh must name the intended live
    /// devices in the same order. XLA additionally validates its build identity, flags, platform, and executable ABI.
    pub fn load<'c>(
        &self,
        domain: &XlaDomain<'c>,
        binding: &XlaKernelCompilerBinding,
        mesh: &DeviceMesh,
    ) -> Result<LoadedKernel<'c>, KernelAotError> {
        if binding.configuration() != self.manifest.configuration {
            return Err(Self::invalid("compiler binding configuration mismatch"));
        }
        let facts = XlaKernelExecutionFacts::from_client(domain.client().map_err(XlaDomainError::from)?, mesh)?;
        if facts.configuration_key()? != self.manifest.execution {
            return Err(Self::invalid("live execution compatibility mismatch"));
        }
        let program = domain
            .deserialize_program(&self.executable)?
            .ok_or_else(|| Self::invalid("the plugin does not support executable deserialization"))?;
        let inputs = self
            .definition
            .operation()
            .input_types()
            .into_iter()
            .map(|value| <&ArrayType>::try_from(&value).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| Self::invalid(error.to_string()))?;
        let outputs = self
            .definition
            .operation()
            .output_types()
            .into_iter()
            .map(|value| <&ArrayType>::try_from(&value).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| Self::invalid(error.to_string()))?;
        if !program.matches_stateless_signature(&inputs, &outputs) || program.mesh() != mesh {
            return Err(Self::invalid("loaded executable boundary differs from the kernel bundle"));
        }
        Ok(LoadedKernel {
            domain: domain.clone(),
            program: Arc::new(program),
            identity: serde_json::to_vec(&(&self.manifest.semantic_digest, &self.manifest.configuration))?,
            execution: self.manifest.execution.clone(),
            effects: self.definition.body().effects().clone(),
        })
    }

    /// Computes the canonical source identity independently of its transport encoding and diagnostic provenance.
    fn semantic_digest(definition: &KernelDefinition) -> Result<String, KernelAotError> {
        Ok(format!(
            "{:x}",
            Sha256::digest(definition.semantic_key().map_err(|error| Self::invalid(error.to_string()))?)
        ))
    }

    /// Constructs an exact bundle-owned diagnostic.
    fn invalid(message: impl Into<String>) -> KernelAotError {
        KernelAotError::Invalid { message: message.into() }
    }
}

/// An AOT-loaded kernel using the existing XLA domain and whole-execution fence.
#[derive(Clone)]
pub struct LoadedKernel<'c> {
    /// Session owning CUDA resources and ordinary invocation/effect policy.
    domain: XlaDomain<'c>,
    /// Existing PJRT executable and its validated invocation metadata.
    program: Arc<XlaCompiledProgram<'c>>,
    /// Exact source and compiler-binding identity used by distributed preflight.
    identity: Vec<u8>,
    /// Live plugin and local topology identity checked by the AOT loader.
    execution: Vec<u8>,
    /// Canonical source effects, retained for stricter functional distributed admission.
    effects: ryft_core::EffectsSummary,
}

impl<'c> LoadedKernel<'c> {
    /// Rejects external effects which cannot be rolled back by discarding functional outputs.
    pub(crate) fn validate_distributed_effects(&self) -> Result<(), KernelAotError> {
        use ryft_core::EffectClass;
        if self.effects.has_explicit_ordered_state() {
            return Err(KernelEmbeddingError::UnsupportedEffect { effect: EffectClass::OrderedState }.into());
        }
        for effect in self.effects.classes() {
            if !matches!(effect, EffectClass::OrderedAssertion | EffectClass::OrderedState) {
                return Err(KernelEmbeddingError::UnsupportedEffect { effect }.into());
            }
        }
        Ok(())
    }

    /// Borrows the existing domain, canonical boundary and identities for host-coordinated execution.
    pub(crate) fn distributed_parts(&self) -> (&XlaDomain<'c>, &[ArrayType], &DeviceMesh, &[u8], &[u8]) {
        (&self.domain, self.program.input_types(), self.program.mesh(), &self.identity, &self.execution)
    }

    /// Submits ordinary array inputs and returns the existing completion-bearing execution handle.
    /// Pending uploads, input retention, aliases, and device failures use XLA's canonical invocation path.
    pub fn call(&self, inputs: Vec<Array<'c>>) -> Result<Execution<Vec<Array<'c>>>, KernelAotError> {
        Ok(self.domain.execute_compiled_async(&self.program, inputs)?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayIrValue, DataType, Device, LogicalMesh, Typed};
    use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::FromPjrt;
    use crate::kernels::staging::tests::binding;
    use crate::kernels::tests::definition;

    use super::*;

    /// Builds transport-only bytes. The opaque executable is never submitted to a native decoder.
    fn bundle() -> KernelAotBundle {
        let definition = definition();
        KernelAotBundle {
            source: serde_json::to_vec(&definition).unwrap(),
            manifest: Manifest {
                configuration: binding(1).configuration().to_vec(),
                execution: b"fixture execution facts".to_vec(),
                semantic_digest: KernelAotBundle::semantic_digest(&definition).unwrap(),
                stable_hlo: "module {}".to_owned(),
                report: serde_json::json!({"unsupported_counter": null}),
            },
            definition,
            executable: b"transport-only executable bytes".to_vec(),
        }
    }

    /// Makes adversarial envelope metadata internally checksummed so structural validation is exercised.
    fn refresh_checksum(bytes: &mut [u8]) {
        let mut hash = Sha256::new();
        hash.update(&bytes[8..32]);
        hash.update(&bytes[64..]);
        bytes[32..64].copy_from_slice(&hash.finalize());
    }

    /// Replaces a manifest while preserving all unrelated sections and the envelope checksum.
    fn replace_manifest(bytes: &mut Vec<u8>, manifest: serde_json::Value) {
        let length = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let replacement = serde_json::to_vec(&manifest).unwrap();
        bytes.splice(64..64 + length, replacement.iter().copied());
        bytes[8..16].copy_from_slice(&(replacement.len() as u64).to_le_bytes());
        refresh_checksum(bytes);
    }

    /// Compiles a real ordinary CPU executable to isolate the AOT loader's stateless boundary checks.
    /// This fixture does not claim execution of a mock adapter's custom call.
    pub(crate) fn executable_bundle<'c>(
        client: &'c Client<'c>,
        domain: &XlaDomain<'c>,
        mesh: &DeviceMesh,
        data_type: DataType,
        capture: bool,
        compiler: &XlaKernelCompilerBinding,
    ) -> KernelAotBundle {
        let r#type = ArrayType::scalar(data_type);
        let captures = if capture {
            vec![ArrayIrValue::Array(
                Array::from_host_buffer(client, r#type.clone(), mesh.clone(), 7i32.to_ne_bytes()).unwrap(),
            )]
        } else {
            vec![]
        };
        let staged = domain
            .stage(CompilationStagingRequest::<XlaDomain<'c>, _, Vec<ArrayIrType>, Vec<ArrayIrType>>::new(
                |_, captures: Vec<CompilationTracer<XlaDomain<'c>>>, inputs| {
                    // Returning the capture retains a real capture slot without introducing unrelated operations.
                    Ok(if captures.is_empty() { inputs } else { captures })
                },
                captures,
                vec![ArrayIrType::Array(r#type)],
                XlaOptions::new(mesh.clone()),
            ))
            .unwrap();
        let compiled = domain.compile(domain.lower(staged).unwrap()).unwrap();
        let mut bundle = bundle();
        bundle.manifest.configuration = compiler.configuration().to_vec();
        bundle.executable = domain.serialize_program(compiled.compiled_program()).unwrap().unwrap();
        bundle.manifest.execution =
            XlaKernelExecutionFacts::from_client(client, mesh).unwrap().configuration_key().unwrap();
        bundle
    }

    #[test]
    fn test_kernel_aot_bundle_compile_missing_binding() {
        let mesh = DeviceMesh::new(LogicalMesh::new(vec![]).unwrap(), vec![Device::new(0, 0)]).unwrap();
        assert!(matches!(
            KernelAotBundle::compile(&definition(), &XlaDomain::clientless(), XlaOptions::new(mesh), 1),
            Err(KernelAotError::Invalid { message }) if message == "a compiler binding is required",
        ));
    }

    #[test]
    fn test_kernel_aot_bundle_compile_without_inputs() {
        use ryft_core::kernels::{Grid, KernelCallOperation};

        let operation = KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |_| Ok(())).unwrap();
        let mesh = DeviceMesh::new(LogicalMesh::new(vec![]).unwrap(), vec![Device::new(0, 0)]).unwrap();
        assert!(matches!(
            KernelAotBundle::compile(&definition, &XlaDomain::clientless(), XlaOptions::new(mesh), 1),
            Err(KernelAotError::Invalid { message })
                if message == "aot compilation requires at least one ordinary kernel input",
        ));
    }

    #[test]
    fn test_kernel_aot_bundle_from_bytes() {
        let original = bundle();
        let bytes = original.to_bytes().unwrap();
        let decoded = KernelAotBundle::from_bytes(&bytes, 1).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        assert_eq!(decoded.definition().semantic_key().unwrap(), original.definition().semantic_key().unwrap());
        assert_eq!(decoded.source, original.source);
        assert_eq!(decoded.executable, original.executable);
        assert_eq!(decoded.stable_hlo(), "module {}");
        assert_eq!(decoded.report(), &serde_json::json!({"unsupported_counter": null}));
    }

    #[test]
    fn test_kernel_aot_bundle_from_bytes_schema_and_checksum() {
        let bytes = bundle().to_bytes().unwrap();
        assert!(matches!(
            KernelAotBundle::from_bytes(&bytes[..63], 1),
            Err(KernelAotError::Invalid { message }) if message == "unsupported schema or bundle size",
        ));
        let mut schema = bytes.clone();
        schema[7] = b'2';
        assert!(matches!(
            KernelAotBundle::from_bytes(&schema, 1),
            Err(KernelAotError::Invalid { message }) if message == "unsupported schema or bundle size",
        ));
        let mut corrupted = bytes;
        *corrupted.last_mut().unwrap() ^= 1;
        assert!(matches!(
            KernelAotBundle::from_bytes(&corrupted, 1),
            Err(KernelAotError::Invalid { message }) if message == "bundle checksum mismatch",
        ));
    }

    #[test]
    fn test_kernel_aot_bundle_from_bytes_section_boundaries() {
        let bytes = bundle().to_bytes().unwrap();
        let mut truncated = bytes.clone();
        truncated.pop().unwrap();
        refresh_checksum(&mut truncated);
        assert!(matches!(
            KernelAotBundle::from_bytes(&truncated, 1),
            Err(KernelAotError::Invalid { message }) if message == "section exceeds the bundle boundary",
        ));
        let mut overflow = bytes.clone();
        overflow[8..16].copy_from_slice(&(usize::MAX as u64).to_le_bytes());
        refresh_checksum(&mut overflow);
        assert!(matches!(
            KernelAotBundle::from_bytes(&overflow, 1),
            Err(KernelAotError::Invalid { message }) if message == "section exceeds the bundle boundary",
        ));
        let mut trailing = bytes;
        trailing.push(0);
        refresh_checksum(&mut trailing);
        assert!(matches!(
            KernelAotBundle::from_bytes(&trailing, 1),
            Err(KernelAotError::Invalid { message }) if message == "empty section or trailing bundle bytes",
        ));
        let mut empty = bundle();
        empty.executable.clear();
        assert!(matches!(
            KernelAotBundle::from_bytes(&empty.to_bytes().unwrap(), 1),
            Err(KernelAotError::Invalid { message }) if message == "empty section or trailing bundle bytes",
        ));
    }

    #[test]
    fn test_kernel_aot_bundle_from_bytes_invalid_metadata() {
        let mut malformed = bundle();
        malformed.source = b"{".to_vec();
        assert!(matches!(
            KernelAotBundle::from_bytes(&malformed.to_bytes().unwrap(), 1),
            Err(KernelAotError::Serialization(error)) if error.is_eof(),
        ));
        let original = bundle();
        let mut bytes = original.to_bytes().unwrap();
        let mut manifest = serde_json::to_value(&original.manifest).unwrap();
        manifest["unknown_schema_field"] = serde_json::json!(true);
        replace_manifest(&mut bytes, manifest);
        assert!(matches!(
            KernelAotBundle::from_bytes(&bytes, 1),
            Err(KernelAotError::Serialization(error)) if error.is_data(),
        ));
        let mut mismatch = bundle();
        mismatch.manifest.semantic_digest = "different semantic identity".to_owned();
        assert!(matches!(
            KernelAotBundle::from_bytes(&mismatch.to_bytes().unwrap(), 1),
            Err(KernelAotError::Invalid { message }) if message == "source semantic identity mismatch",
        ));
    }

    #[test]
    fn test_kernel_aot_bundle_to_bytes_size_limit() {
        let mut oversized = bundle();
        oversized.source.resize(KernelAotBundle::MAXIMUM_JSON_BYTES + 1, 0);
        assert!(matches!(
            oversized.to_bytes(),
            Err(KernelAotError::Invalid { message }) if message == "JSON section exceeds the size limit",
        ));
    }

    #[test]
    fn test_kernel_aot_bundle_load_configuration() {
        let mesh = DeviceMesh::new(LogicalMesh::new(vec![]).unwrap(), vec![Device::new(0, 0)]).unwrap();
        assert!(matches!(
            bundle().load(&XlaDomain::clientless(), &binding(2), &mesh),
            Err(KernelAotError::Invalid { message }) if message == "compiler binding configuration mismatch",
        ));
    }

    #[test]
    fn test_kernel_aot_bundle_load_live_facts() {
        let client = crate::tests::execution_client();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![]).unwrap(),
            vec![Device::from_pjrt(client.addressable_devices().unwrap().remove(0)).unwrap()],
        )
        .unwrap();
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        assert!(matches!(
            bundle().load(&domain, &binding(1), &mesh),
            Err(KernelAotError::Invalid { message }) if message == "live execution compatibility mismatch",
        ));
    }

    #[test]
    fn test_kernel_aot_bundle_load_stateless_boundary() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![]).unwrap(),
            vec![Device::from_pjrt(client.addressable_devices().unwrap().remove(0)).unwrap()],
        )
        .unwrap();
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let original = executable_bundle(&client, &domain, &mesh, DataType::I32, false, &binding(1));
        let decoded = KernelAotBundle::from_bytes(&original.to_bytes().unwrap(), 1).unwrap();
        let loaded = decoded.load(&domain, &binding(1), &mesh).unwrap();
        let input =
            Array::from_host_buffer(&client, ArrayType::scalar(DataType::I32), mesh.clone(), 23i32.to_ne_bytes())
                .unwrap();
        let execution = loaded.call(vec![input]).unwrap();
        execution.fence().block_until_ready().unwrap();
        assert_eq!(execution.output().len(), 1);
        assert_eq!(execution.output()[0].r#type().data_type(), DataType::I32);
        let bytes = execution.output()[0]
            .addressable_shards()
            .next()
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(i32::from_ne_bytes(bytes.as_slice().try_into().unwrap()), 23);

        let expected = "loaded executable boundary differs from the kernel bundle";
        let wrong_input = executable_bundle(&client, &domain, &mesh, DataType::F32, false, &binding(1));
        assert!(matches!(
            wrong_input.load(&domain, &binding(1), &mesh),
            Err(KernelAotError::Invalid { message }) if message == expected,
        ));
        let captured = executable_bundle(&client, &domain, &mesh, DataType::I32, true, &binding(1));
        assert!(matches!(
            captured.load(&domain, &binding(1), &mesh),
            Err(KernelAotError::Invalid { message }) if message == expected,
        ));
    }
    #[test]
    fn test_loaded_kernel_validate_distributed_effects() {
        use ryft_core::{EffectClass, EffectClasses, Effects};

        let client = crate::tests::execution_client();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![]).unwrap(),
            vec![Device::from_pjrt(client.addressable_devices().unwrap().remove(0)).unwrap()],
        )
        .unwrap();
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let bundle = executable_bundle(&client, &domain, &mesh, DataType::I32, false, &binding(1));
        let mut loaded = bundle.load(&domain, &binding(1), &mesh).unwrap();
        assert!(loaded.validate_distributed_effects().is_ok());
        for effect in
            [EffectClass::OrderedIo, EffectClass::DeviceOrderedIo, EffectClass::UnorderedIo, EffectClass::OrderedState]
        {
            loaded.effects = Effects::new(EffectClasses::single(effect), vec![], vec![]).unwrap().summary();
            assert!(matches!(
                loaded.validate_distributed_effects(),
                Err(KernelAotError::Embedding(KernelEmbeddingError::UnsupportedEffect { effect: actual }))
                    if actual == effect,
            ));
        }
        loaded.effects = Effects::new(EffectClasses::single(EffectClass::OrderedAssertion), vec![], vec![])
            .unwrap()
            .summary();
        assert!(loaded.validate_distributed_effects().is_ok());
    }
}
