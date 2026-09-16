//! In-process native compilation and concrete device artifact validation.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ryft_core::kernels::{
    KERNEL_CALL_OPERATION_NAME, KernelCompilationError, KernelCompiler, KernelSchedule, VerifiedKernel,
};
use ryft_core::{EffectClass, Typed};
use ryft_cuda::{
    CudaArtifactFormat, CudaKernelAbi, CudaKernelArtifact, CudaKernelLaunchDimensions, CudaKernelParameterType,
};
use ryft_mlir::{Context, StringRef};
use ryft_rocm::{RocmKernelArtifact, RocmKernelLaunchDimensions};
use ryft_xla_sys::triton::{
    RYFT_XLA_Triton_Compile, RYFT_XLA_Triton_Compile_Args, RYFT_XLA_Triton_Compile_Args_Destroy,
    RYFT_XLA_Triton_Get_Versions, RYFT_XLA_Triton_Versions, RYFT_XLA_Triton_Versions_Destroy,
};
use serde_json::{Value, json};

use crate::kernels::lowering;
use crate::kernels::{
    Artifact, COMPILER_SCHEMA_VERSION, CompiledKernel, Error, Options, TRITON_VERSION, Target, XLA_VERSION,
};

/// Linked native compiler and optional caller-owned cancellation signal.
///
/// Compilation has the same in-process failure model as ordinary XLA compilation. Recoverable native failures
/// return diagnostics. Cancellation is observed before and after the native call; it cannot interrupt that call
/// or contain native aborts. Recorded configuration supports AOT compatibility checks without compiling.
#[derive(Clone, Debug, Default)]
pub struct Compiler {
    /// Recorded configuration disables compilation while retaining exact AOT compatibility checks.
    configuration: Option<Value>,

    /// Cancellation controls submission and publication, not semantic identity.
    cancellation: Arc<AtomicBool>,
}

/// Checked entry metadata used to validate concrete PTX declarations.
struct Metadata {
    /// Number of ordinary global pointer arguments.
    argument_count: usize,

    /// Actual thread block width after native specialization.
    block_dimension_x: u32,
}

/// Owns native compilation outputs through every error and cancellation path.
struct NativeCompilation(RYFT_XLA_Triton_Compile_Args);

impl Drop for NativeCompilation {
    fn drop(&mut self) {
        // The bridge releases only its owned outputs; the input module remains owned by the caller.
        unsafe { RYFT_XLA_Triton_Compile_Args_Destroy(&mut self.0) };
    }
}

/// Owns the assembler version returned alongside process-lifetime source revision strings.
struct NativeVersions(RYFT_XLA_Triton_Versions);

impl Drop for NativeVersions {
    fn drop(&mut self) {
        unsafe { RYFT_XLA_Triton_Versions_Destroy(&mut self.0) };
    }
}

impl Compiler {
    /// Selects the native bridge linked into `ryft-xla-sys` without opening a GPU context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Restores validated compiler configuration for AOT loading without invoking the compiler or a GPU runtime.
    /// This compiler can check compatibility but cannot compile new kernels. Configuration records describe
    /// compatibility; they do not authenticate native code or its producer.
    pub fn from_configuration(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() > 1024 * 1024 {
            return Err(Error::Invalid { message: "compiler configuration exceeds 1 MiB".into() });
        }
        let configuration: Value = serde_json::from_slice(bytes)?;
        let fields = configuration
            .as_object()
            .ok_or_else(|| Error::Invalid { message: "compiler configuration must be an object".into() })?;
        let expected = ["schema", "versions", "target", "warp_count", "pipeline_stages", "maximum_scratch_bytes"];
        if fields.len() != expected.len()
            || expected.iter().any(|name| !fields.contains_key(*name))
            || configuration["schema"] != json!(COMPILER_SCHEMA_VERSION)
        {
            return Err(Error::Invalid { message: "invalid recorded compiler configuration".into() });
        }
        let target: Target = serde_json::from_value(configuration["target"].clone())?;
        target.validate()?;
        Self::validate_versions(&configuration["versions"], &target)?;
        let warps = configuration["warp_count"].as_u64().unwrap_or(0);
        let stages = configuration["pipeline_stages"].as_u64().unwrap_or(0);
        if !matches!(warps, 1 | 2 | 4 | 8)
            || !(1..=8).contains(&stages)
            || (!configuration["maximum_scratch_bytes"].is_null()
                && configuration["maximum_scratch_bytes"].as_u64().is_none())
        {
            return Err(Error::Invalid { message: "invalid recorded compiler options".into() });
        }
        Ok(Self { configuration: Some(configuration), ..Self::new() })
    }

    /// Uses a caller-owned signal checked before native compilation and before publishing its result.
    /// An in-progress native call runs to completion even when cancellation is requested.
    pub fn with_cancellation(mut self, cancellation: Arc<AtomicBool>) -> Self {
        self.cancellation = cancellation;
        self
    }

    /// Computes linked compiler and effective toolchain identity, or checks the recorded AOT configuration.
    fn key(&self, target: &Target, options: &Options, schedule: &KernelSchedule) -> Result<Vec<u8>, Error> {
        target.validate()?;
        let stages = schedule.pipeline_stages().map_or(2, |stages| stages.get());
        if stages > 8 || schedule.buffering_depth().is_some() {
            return Err(Error::Invalid {
                message: "Triton supports at most eight pipeline stages and no buffering-depth override".into(),
            });
        }
        if let Some(configuration) = &self.configuration {
            if configuration["target"] != serde_json::to_value(target)?
                || configuration["warp_count"] != json!(options.warp_count())
                || configuration["pipeline_stages"] != json!(stages)
                || configuration["maximum_scratch_bytes"] != json!(schedule.maximum_scratch_bytes())
            {
                return Err(Error::Invalid {
                    message: "requested options differ from recorded compiler configuration".into(),
                });
            }
            return Ok(serde_json::to_vec(configuration)?);
        }
        let native = NativeVersions(unsafe { RYFT_XLA_Triton_Get_Versions() });
        let available = match target {
            Target::Cuda { .. } => native.0.cuda_available,
            Target::Rocm { .. } => native.0.rocm_available,
        };
        if !available {
            return Err(Error::Unavailable {
                message: "linked native archive does not include the requested Triton compiler target".into(),
            });
        }
        let versions = json!({
            "schema": COMPILER_SCHEMA_VERSION,
            "xla": unsafe { StringRef::from_c_api(native.0.xla) }.to_string(),
            "jax": unsafe { StringRef::from_c_api(native.0.jax) }.to_string(),
            "triton": unsafe { StringRef::from_c_api(native.0.triton) }.to_string(),
            "rocm_device_libs": unsafe { StringRef::from_c_api(native.0.rocm_device_libs) }.to_string(),
            "cuda_available": native.0.cuda_available,
            "rocm_available": native.0.rocm_available,
            "cuda_toolkit_version": native.0.cuda_toolkit_version,
            "assembler_version": String::from_utf8(unsafe {
                copy_native_bytes(native.0.assembler_version, native.0.assembler_version_size, 1024)?
            }).map_err(|_| Error::Invalid { message: "native assembler version is not UTF-8".into() })?,
        });
        Self::validate_versions(&versions, target)?;
        Ok(serde_json::to_vec(&json!({
            "schema": COMPILER_SCHEMA_VERSION, "versions": versions, "target": target,
            "warp_count": options.warp_count(), "pipeline_stages": stages,
            "maximum_scratch_bytes": schedule.maximum_scratch_bytes(),
        }))?)
    }

    /// Checks the same linked-source and target toolchain contract for live and recorded configurations.
    fn validate_versions(versions: &Value, target: &Target) -> Result<(), Error> {
        let fixed = json!({
            "schema": COMPILER_SCHEMA_VERSION, "xla": XLA_VERSION,
            "jax": "a7606f995e1a92707cbeb257e487fa53e7abe84b", "triton": TRITON_VERSION,
            "rocm_device_libs": "53996464fa8d94b182ac4aaa7dc3a109ab524f45",
        });
        let valid_toolchain = match target {
            Target::Cuda { .. } => {
                versions["cuda_available"] == json!(true)
                    && versions["cuda_toolkit_version"] == json!(13020)
                    && versions["assembler_version"] == json!("13.0.88")
            }
            Target::Rocm { .. } => {
                versions["rocm_available"] == json!(true)
                    && versions["cuda_toolkit_version"]
                        .as_u64()
                        .is_some_and(|version| version == 0 || (version >= 12000 && version % 10 == 0))
                    && versions["assembler_version"].as_str().is_some_and(|version| {
                        version == "unavailable"
                            || (version.split('.').count() == 3
                                && version
                                    .split('.')
                                    .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit())))
                    })
            }
        };
        if !versions.as_object().is_some_and(|fields| fields.len() == 9)
            || fixed.as_object().unwrap().iter().any(|(name, value)| versions[name] != *value)
            || versions["cuda_available"].as_bool().is_none()
            || versions["rocm_available"].as_bool().is_none()
            || !valid_toolchain
        {
            return Err(Error::Invalid {
                message: "compiler installation differs from the qualified source and toolchain contract".into(),
            });
        }
        Ok(())
    }

    /// Rejects unsupported physical buffers and target-specific effects before native compilation.
    fn validate_boundary(kernel: &VerifiedKernel<'_>, target: &Target) -> Result<(), Error> {
        if matches!(target, Target::Rocm { .. })
            && !(1..=64).contains(&kernel.definition().operation().parameters().len())
        {
            return Err(Error::Unsupported {
                operation: KERNEL_CALL_OPERATION_NAME,
                reason: "ROCm requires between one and 64 physical pointer parameters".into(),
            });
        }
        if kernel
            .definition()
            .operation()
            .parameters()
            .iter()
            .any(|parameter| parameter.r#type().static_shape().is_some_and(|shape| shape.dimensions().contains(&0)))
        {
            return Err(Error::Unsupported {
                operation: "array",
                reason: "zero-element physical parameters require an unqualified device pointer contract".into(),
            });
        }
        if matches!(target, Target::Rocm { .. })
            && kernel.definition().body().effects().classes().contains(EffectClass::OrderedAssertion)
        {
            return Err(Error::Unsupported {
                operation: "ordered_assertion",
                reason: "AMD assertion hostcall services have not been qualified".into(),
            });
        }
        Ok(())
    }
}

impl KernelCompiler for Compiler {
    type Target = Target;
    type Options = Options;
    type Output = CompiledKernel;
    type Error = Error;

    fn admit(
        &self,
        kernel: &VerifiedKernel<'_>,
        target: &Target,
        options: &Options,
        schedule: &KernelSchedule,
    ) -> Result<(), KernelCompilationError<Error>> {
        target.validate().map_err(classify)?;
        Self::validate_boundary(kernel, target).map_err(classify)?;
        if self.cancellation.load(Ordering::Acquire) {
            return Err(classify(Error::Cancelled));
        }
        let context = Context::new();
        lowering::lower(&context, kernel, options, schedule).map_err(classify)?;
        self.key(target, options, schedule).map_err(classify)?;
        Ok(())
    }

    fn configuration_key(
        &self,
        target: &Target,
        options: &Options,
        schedule: &KernelSchedule,
    ) -> Result<Vec<u8>, KernelCompilationError<Error>> {
        self.key(target, options, schedule).map_err(classify)
    }

    fn compile(
        &self,
        kernel: &VerifiedKernel<'_>,
        target: &Target,
        options: &Options,
        schedule: &KernelSchedule,
    ) -> Result<CompiledKernel, KernelCompilationError<Error>> {
        let compile = || -> Result<CompiledKernel, Error> {
            target.validate()?;
            Self::validate_boundary(kernel, target)?;
            if self.configuration.is_some() {
                return Err(Error::Invalid {
                    message: "recorded compiler configuration cannot compile new kernels".into(),
                });
            }
            if self.cancellation.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            let context = Context::new();
            let lowered = lowering::lower(&context, kernel, options, schedule)?;
            let configuration_key = self.key(target, options, schedule)?;
            let (platform, architecture, threads_per_warp) = match target {
                Target::Cuda { major, minor } => ("cuda", format!("{major}.{minor}"), 32),
                Target::Rocm { architecture } => ("rocm", architecture.clone(), 64),
            };
            let mut native = NativeCompilation(RYFT_XLA_Triton_Compile_Args::new(
                unsafe { lowered.module.to_c_api() },
                unsafe { StringRef::from(platform).to_c_api() },
                unsafe { StringRef::from(architecture.as_str()).to_c_api() },
                options.warp_count() as i32,
                schedule.pipeline_stages().map_or(2, |stages| stages.get()) as i32,
                options.maximum_artifact_bytes(),
                options.maximum_diagnostic_bytes(),
            ));
            if self.cancellation.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            let success = {
                // Native passes mutate a clone in this context. Keep the exclusive guard through the entire call.
                let _guard = context.borrow_mut();
                unsafe { RYFT_XLA_Triton_Compile(&mut native.0).value != 0 }
            };
            if self.cancellation.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            let diagnostics = String::from_utf8_lossy(&unsafe {
                copy_native_bytes(native.0.diagnostics, native.0.diagnostics_size, options.maximum_diagnostic_bytes())?
            })
            .into_owned();
            if !success {
                return Err(Error::Compilation { message: diagnostics });
            }
            let entry = unsafe { copy_native_bytes(native.0.entry_name, native.0.entry_name_size, 1024)? };
            if entry != b"ryft_kernel"
                || native.0.argument_count != lowered.parameter_types.len() as i64
                || native.0.threads_per_warp != threads_per_warp
                || native.0.actual_warp_count <= 0
                || native.0.actual_warp_count > 1024 / threads_per_warp
                || !(0..=1024 * 1024).contains(&native.0.shared_memory_bytes)
                || schedule.maximum_scratch_bytes().is_some_and(|limit| native.0.shared_memory_bytes as usize > limit)
            {
                return Err(Error::Artifact {
                    message: "native metadata differs from the requested entry, target or resource contract".into(),
                });
            }
            let metadata = Metadata {
                argument_count: lowered.parameter_types.len(),
                block_dimension_x: (native.0.actual_warp_count * threads_per_warp) as u32,
            };
            let shared_memory_bytes = native.0.shared_memory_bytes as u32;
            let bytes = unsafe {
                copy_native_bytes(native.0.artifact, native.0.artifact_size, options.maximum_artifact_bytes())?
            };
            let artifact = match target {
                Target::Cuda { .. } => {
                    let ptx = std::str::from_utf8(&bytes)
                        .map_err(|_| Error::Artifact { message: "PTX is not UTF-8".into() })?;
                    let architecture = validate_ptx(ptx, target, &metadata)?;
                    Artifact::Cuda(CudaKernelArtifact::new(
                        CudaArtifactFormat::Ptx,
                        bytes,
                        "ryft_kernel",
                        architecture,
                        CudaKernelLaunchDimensions::new(
                            lowered.grid,
                            [metadata.block_dimension_x, 1, 1],
                            shared_memory_bytes,
                        )?,
                        CudaKernelAbi::new(
                            "triton global pointers",
                            1,
                            vec![CudaKernelParameterType::DevicePointer; metadata.argument_count],
                        )?,
                    )?)
                }
                Target::Rocm { architecture } => Artifact::Rocm(RocmKernelArtifact::new(
                    bytes.into(),
                    "ryft_kernel",
                    architecture,
                    metadata.argument_count,
                    RocmKernelLaunchDimensions::new(
                        lowered.grid,
                        [metadata.block_dimension_x, 1, 1],
                        shared_memory_bytes,
                    )?,
                )?),
            };
            if self.key(target, options, schedule)? != configuration_key {
                return Err(Error::Invalid { message: "compiler toolchain changed during compilation".into() });
            }
            if self.cancellation.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            Ok(CompiledKernel {
                artifact,
                parameter_types: lowered.parameter_types,
                semantic_key: kernel.definition().semantic_key()?,
                configuration_key,
                diagnostics,
            })
        };
        compile().map_err(classify)
    }
}

/// Copies a bounded native output while its owning result remains alive.
///
/// # Safety
/// A nonempty buffer must be readable for its declared size until the copy completes. The native bridge guarantees
/// this for its owned result buffers. Empty results may use a null pointer.
unsafe fn copy_native_bytes(pointer: *const u8, size: usize, maximum: usize) -> Result<Vec<u8>, Error> {
    if size > maximum || (size != 0 && pointer.is_null()) {
        return Err(Error::Artifact { message: "native output is null or exceeds its size limit".into() });
    }
    if size == 0 {
        return Ok(Vec::new());
    }
    Ok(unsafe { std::slice::from_raw_parts(pointer, size) }.to_vec())
}

/// Preserves canonical admission categories and concrete native diagnostics.
fn classify(error: Error) -> KernelCompilationError<Error> {
    match error {
        Error::Unavailable { message } => KernelCompilationError::Unavailable { message },
        Error::Unsupported { operation, reason } => KernelCompilationError::Unsupported {
            owner: "ryft_triton::kernels".into(),
            operation,
            requested: reason,
            capability: "direct Triton lowering".into(),
        },
        error => KernelCompilationError::Compiler(error),
    }
}

/// Tokenizes PTX declarations while excluding comments and quoted strings from ABI inspection.
fn tokens(source: &str) -> Result<Vec<&str>, Error> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if source[index..].starts_with("//") {
            index += source[index..].find('\n').unwrap_or(bytes.len() - index);
            continue;
        }
        if source[index..].starts_with("/*") {
            index += source[index + 2..]
                .find("*/")
                .ok_or_else(|| Error::Artifact { message: "unterminated PTX comment".into() })?
                + 4;
            continue;
        }
        if bytes[index] == b'"' {
            index += 1;
            loop {
                if index >= bytes.len() {
                    return Err(Error::Artifact { message: "unterminated PTX string".into() });
                }
                if bytes[index] == b'"' {
                    index += 1;
                    break;
                }
                index += if bytes[index] == b'\\' { 2 } else { 1 };
            }
            continue;
        }
        let start = index;
        index += 1;
        if !b"(){},;".contains(&bytes[start]) {
            while index < bytes.len() && !bytes[index].is_ascii_whitespace() && !b"(){},;\"".contains(&bytes[index]) {
                index += 1;
            }
        }
        tokens.push(&source[start..index]);
    }
    Ok(tokens)
}

/// Requires the actual PTX declaration and resources to agree with the post-LLVM native signature.
fn validate_ptx(source: &str, target: &Target, metadata: &Metadata) -> Result<String, Error> {
    let tokens = tokens(source)?;
    let invalid = || Error::Artifact { message: "PTX declarations differ from the native artifact contract".into() };
    let unique = |name: &str| -> Result<usize, Error> {
        let mut positions = tokens.iter().enumerate().filter_map(|(index, token)| (*token == name).then_some(index));
        let position = positions.next().ok_or_else(invalid)?;
        if positions.next().is_some() {
            return Err(invalid());
        }
        Ok(position)
    };
    if tokens.get(unique(".version")? + 1) != Some(&"9.0") || tokens.get(unique(".address_size")? + 1) != Some(&"64") {
        return Err(invalid());
    }
    let Target::Cuda { major, minor } = target else {
        return Err(invalid());
    };
    let architecture = format!("sm_{major}{minor}");
    if tokens.get(unique(".target")? + 1).copied() != Some(architecture.as_str()) {
        return Err(invalid());
    }
    let mut index = unique(".entry")? + 1;
    if tokens.get(index) != Some(&"ryft_kernel") || tokens.get(index + 1) != Some(&"(") {
        return Err(invalid());
    }
    index += 2;
    for parameter in 0..metadata.argument_count {
        if tokens.get(index..index + 5) != Some(&[".param", ".u64", ".ptr", ".global", ".align"][..])
            || tokens.get(index + 5) != Some(&"1")
        {
            return Err(invalid());
        }
        let name = tokens.get(index + 6).ok_or_else(invalid)?;
        if name.is_empty() || !name.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"_$".contains(&byte)) {
            return Err(invalid());
        }
        index += 7;
        if parameter + 1 != metadata.argument_count {
            if tokens.get(index) != Some(&",") {
                return Err(invalid());
            }
            index += 1;
        }
    }
    if tokens.get(index) != Some(&")") || tokens.get(index + 1) != Some(&".reqntid") {
        return Err(invalid());
    }
    index += 2;
    let width = metadata.block_dimension_x.to_string();
    if tokens.get(index).copied() != Some(width.as_str()) {
        return Err(invalid());
    }
    index += 1;
    if tokens.get(index) == Some(&",") {
        if tokens.get(index..index + 4) != Some(&[",", "1", ",", "1"][..]) {
            return Err(invalid());
        }
        index += 4;
    }
    if tokens.get(index) != Some(&"{") {
        return Err(invalid());
    }
    Ok(architecture)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::kernels::{KernelCallOperation, KernelDefinition};
    use ryft_core::{ArrayType, DataType};

    use super::*;

    /// A complete PTX entry declaration matching the native pointer-only contract.
    const PTX: &str = indoc! {"
        .version 9.0
        .target sm_121
        .address_size 64
        .visible .entry ryft_kernel(
            .param .u64 .ptr .global .align 1 pointer
        )
        .reqntid 128, 1, 1
        { ret; }
    "};

    /// Metadata for a single-pointer, zero-scratch native program.
    fn metadata() -> Metadata {
        Metadata { argument_count: 1, block_dimension_x: 128 }
    }

    /// A portable definition used to exercise the native compilation without changing its portable semantics.
    #[ryft_core::kernels::kernel]
    fn vector(
        #[input(data_type = F32, rank = 1)] left: &Array,
        #[input(data_type = F32, rank = 1)] right: &Array,
        #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut Array,
    ) {
        output.store(left.load() + right.load());
    }

    /// Bounded index conversion preserves its ordered assertion even before native lowering.
    #[ryft_core::kernels::kernel]
    fn matrix(
        #[input(data_type = F32, rank = 2)] left: &Array,
        #[input(data_type = F32, rank = 2)] right: &Array,
        #[output(data_type = F32, shape = [32, 32])] output: &mut Array,
    ) {
        let left_tiles = left.tiles([32, 32]).pad(0.0);
        let right_tiles = right.tiles([32, 32]).pad(0.0);
        let mut accumulator = zeros::<f32>([32, 32]);
        for depth in 0..left.shape()[1].div_ceil(32) {
            accumulator += left_tiles.load([0, depth]).dot(right_tiles.load([depth, 0]));
        }
        output.store(accumulator);
    }

    /// Produces the fixed source contract used by both the native and recorded paths.
    fn versions() -> Value {
        json!({
            "schema": COMPILER_SCHEMA_VERSION, "xla": XLA_VERSION,
            "jax": "a7606f995e1a92707cbeb257e487fa53e7abe84b", "triton": TRITON_VERSION,
            "rocm_device_libs": "53996464fa8d94b182ac4aaa7dc3a109ab524f45",
            "cuda_available": true, "rocm_available": true,
            "cuda_toolkit_version": 13020, "assembler_version": "13.0.88",
        })
    }

    #[test]
    fn test_compiler_new() {
        let compiler = Compiler::new();
        assert!(compiler.configuration.is_none());
        assert!(!compiler.cancellation.load(Ordering::Acquire));
    }

    #[test]
    fn test_compiler_from_configuration() {
        let configuration = json!({
            "schema": COMPILER_SCHEMA_VERSION, "versions": versions(),
            "target": Target::Cuda { major: 12, minor: 1 }, "warp_count": 4,
            "pipeline_stages": 2, "maximum_scratch_bytes": null,
        });
        let bytes = serde_json::to_vec(&configuration).unwrap();
        let restored = Compiler::from_configuration(&bytes).unwrap();
        assert!(restored.configuration.is_some());
        assert_eq!(
            restored
                .key(&Target::Cuda { major: 12, minor: 1 }, &Options::default(), &KernelSchedule::default())
                .unwrap(),
            bytes
        );
        for field in ["jax", "triton", "rocm_device_libs", "assembler_version", "cuda_toolkit_version", "unexpected"] {
            let mut changed = configuration.clone();
            changed["versions"][field] = json!("unqualified");
            assert!(matches!(
                Compiler::from_configuration(&serde_json::to_vec(&changed).unwrap()),
                Err(Error::Invalid { message })
                    if message == "compiler installation differs from the qualified source and toolchain contract"
            ));
        }
        for (field, value) in
            [("warp_count", json!(3)), ("pipeline_stages", json!(0)), ("maximum_scratch_bytes", json!(-1))]
        {
            let mut changed = configuration.clone();
            changed[field] = value;
            assert!(matches!(Compiler::from_configuration(&serde_json::to_vec(&changed).unwrap()),
                Err(Error::Invalid { message }) if message == "invalid recorded compiler options"));
        }
        let mut rocm = configuration.clone();
        rocm["target"] = json!(Target::Rocm { architecture: "gfx942".into() });
        rocm["versions"]["cuda_toolkit_version"] = json!(0);
        rocm["versions"]["assembler_version"] = json!("unavailable");
        assert!(Compiler::from_configuration(&serde_json::to_vec(&rocm).unwrap()).is_ok());
        assert!(matches!(Compiler::from_configuration(&vec![b' '; 1024 * 1024 + 1]),
            Err(Error::Invalid { message }) if message == "compiler configuration exceeds 1 MiB"));
        assert!(matches!(Compiler::from_configuration(b"[]"), Err(Error::Invalid { message })
            if message == "compiler configuration must be an object"));
    }

    #[test]
    fn test_compiler_with_cancellation() {
        let cancellation = Arc::new(AtomicBool::new(true));
        let compiler = Compiler::new().with_cancellation(cancellation);
        let r#type = ArrayType::new_static(DataType::F32, [256]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let kernel = VerifiedKernel::new(&definition, 1024).unwrap();
        let target = Target::Cuda { major: 12, minor: 1 };
        assert!(matches!(
            compiler.admit(&kernel, &target, &Options::default(), &KernelSchedule::default()),
            Err(KernelCompilationError::Compiler(Error::Cancelled))
        ));
        assert!(matches!(
            compiler.compile(&kernel, &target, &Options::default(), &KernelSchedule::default()),
            Err(KernelCompilationError::Compiler(Error::Cancelled))
        ));
    }

    #[test]
    fn test_compiler_configuration_key() {
        let configuration = json!({
            "schema": COMPILER_SCHEMA_VERSION, "versions": versions(), "target": Target::Cuda { major: 12, minor: 1 },
            "warp_count": 4, "pipeline_stages": 2, "maximum_scratch_bytes": null,
        });
        let bytes = serde_json::to_vec(&configuration).unwrap();
        let compiler = Compiler::from_configuration(&bytes).unwrap();
        let target = Target::Cuda { major: 12, minor: 1 };
        assert_eq!(
            compiler.configuration_key(&target, &Options::default(), &KernelSchedule::default()).unwrap(),
            bytes
        );
        assert!(
            matches!(compiler.key(&target, &Options::default().with_warp_count(8).unwrap(), &KernelSchedule::default()),
            Err(Error::Invalid { message }) if message == "requested options differ from recorded compiler configuration")
        );
        let mut old = configuration.clone();
        old["schema"] = json!(1);
        assert!(matches!(Compiler::from_configuration(&serde_json::to_vec(&old).unwrap()),
            Err(Error::Invalid { message }) if message == "invalid recorded compiler configuration"));
    }

    #[test]
    fn test_compiler_admit() {
        let compiler = Compiler::new();
        let r#type = ArrayType::new_static(DataType::F32, [32, 32]);
        let definition = matrix::definition(&r#type, &r#type).unwrap();
        let verified = VerifiedKernel::new(&definition, 1024).unwrap();
        assert!(matches!(
            compiler.admit(
                &verified,
                &Target::Rocm { architecture: "gfx942".into() },
                &Options::default(),
                &KernelSchedule::default()
            ),
            Err(KernelCompilationError::Unsupported { operation: "ordered_assertion", .. })
        ));
    }

    #[test]
    fn test_compiler_admit_and_compile_empty_parameters() {
        let compiler = Compiler::new();
        let r#type = ArrayType::new_static(DataType::F32, [0]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let verified = VerifiedKernel::new(&definition, 1024).unwrap();
        let options = Options::default();
        let schedule = KernelSchedule::default();
        for target in [Target::Cuda { major: 12, minor: 1 }, Target::Rocm { architecture: "gfx942".into() }] {
            for result in [
                compiler.admit(&verified, &target, &options, &schedule),
                compiler.compile(&verified, &target, &options, &schedule).map(|_| ()),
            ] {
                assert!(matches!(
                    result,
                    Err(KernelCompilationError::Unsupported { owner, operation, requested, capability })
                        if owner == "ryft_triton::kernels" && operation == "array"
                            && requested == "zero-element physical parameters require an unqualified \
                                             device pointer contract"
                            && capability == "direct Triton lowering"
                ));
            }
        }
    }

    #[test]
    fn test_compiler_admit_and_compile_rocm_parameter_count() {
        let compiler = Compiler::new();
        let r#type = ArrayType::new_static(DataType::F32, [1]);
        let prototype = vector::definition(&r#type, &r#type).unwrap();
        let options = Options::default();
        let schedule = KernelSchedule::default();
        let target = Target::Rocm { architecture: "gfx942".into() };
        for count in [0, 64, 65] {
            let operation = KernelCallOperation::new(
                prototype.operation().grid().clone(),
                vec![prototype.operation().parameters()[0].clone(); count],
            )
            .unwrap();
            let definition: KernelDefinition = KernelDefinition::trace(operation, |_| Ok(())).unwrap();
            let verified = VerifiedKernel::new(&definition, 1024).unwrap();
            assert!(Compiler::validate_boundary(&verified, &Target::Cuda { major: 12, minor: 1 }).is_ok());
            if count == 64 {
                assert!(Compiler::validate_boundary(&verified, &target).is_ok());
                continue;
            }
            for result in [
                compiler.admit(&verified, &target, &options, &schedule),
                compiler.compile(&verified, &target, &options, &schedule).map(|_| ()),
            ] {
                assert!(matches!(
                    result,
                    Err(KernelCompilationError::Unsupported { owner, operation, requested, capability })
                        if owner == "ryft_triton::kernels" && operation == "kernel_call"
                            && requested == "ROCm requires between one and 64 physical pointer parameters"
                            && capability == "direct Triton lowering"
                ));
            }
        }
    }

    #[test]
    fn test_compiler_compile() {
        let configuration = json!({
            "schema": COMPILER_SCHEMA_VERSION, "versions": versions(), "target": Target::Cuda { major: 12, minor: 1 },
            "warp_count": 4, "pipeline_stages": 2, "maximum_scratch_bytes": null,
        });
        let compiler = Compiler::from_configuration(&serde_json::to_vec(&configuration).unwrap()).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [256]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let kernel = VerifiedKernel::new(&definition, 1024).unwrap();
        assert!(matches!(compiler.compile(&kernel, &Target::Cuda { major: 12, minor: 1 },
            &Options::default(), &KernelSchedule::default()),
            Err(KernelCompilationError::Compiler(Error::Invalid { message }))
                if message == "recorded compiler configuration cannot compile new kernels"));
    }

    #[test]
    fn test_compiler_compile_native() {
        if std::env::var("RYFT_RUN_TRITON_COMPILER_TESTS").ok().as_deref() != Some("1") {
            return;
        }
        let compiler = Compiler::new();
        let r#type = ArrayType::new_static(DataType::F32, [256]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let original = definition.body().to_string();
        let kernel = VerifiedKernel::new(&definition, 1024).unwrap();
        let options = Options::default();
        let schedule = KernelSchedule::default();
        for target in [
            Target::Cuda { major: 8, minor: 0 },
            Target::Cuda { major: 12, minor: 1 },
            Target::Rocm { architecture: "gfx942".into() },
        ] {
            compiler.admit(&kernel, &target, &options, &schedule).unwrap();
            let configuration = compiler.configuration_key(&target, &options, &schedule).unwrap();
            let output = compiler.compile(&kernel, &target, &options, &schedule).unwrap();
            assert_eq!(output.semantic_key(), definition.semantic_key().unwrap());
            assert_eq!(output.parameter_types(), vec![r#type.clone(); 3]);
            assert_eq!(output.configuration_key(), configuration);
            assert_eq!(definition.body().to_string(), original);
            assert!(output.diagnostics().len() <= options.maximum_diagnostic_bytes());
            match (target.clone(), output.artifact()) {
                (Target::Cuda { major, minor }, Artifact::Cuda(artifact)) => {
                    assert_eq!(artifact.symbol(), "ryft_kernel");
                    assert_eq!(artifact.target_architecture(), format!("sm_{major}{minor}"));
                    assert_eq!(artifact.launch_dimensions().grid(), [1, 1, 1]);
                    assert_eq!(artifact.launch_dimensions().block(), [128, 1, 1]);
                    assert!(!artifact.bytes().is_empty());
                    assert!(artifact.bytes().len() <= options.maximum_artifact_bytes());
                }
                (Target::Rocm { .. }, Artifact::Rocm(artifact)) => {
                    assert_eq!(artifact.entry_name(), "ryft_kernel");
                    assert_eq!(artifact.target(), "gfx942");
                    assert_eq!(artifact.parameter_count(), 3);
                    assert_eq!(artifact.launch_dimensions().grid(), [1, 1, 1]);
                    assert_eq!(artifact.launch_dimensions().block(), [256, 1, 1]);
                    assert!(!artifact.image().is_empty());
                    assert!(artifact.image().len() <= options.maximum_artifact_bytes());
                }
                _ => panic!("native compiler returned an artifact for a different platform"),
            }
            let deployment = Compiler::from_configuration(&configuration).unwrap();
            assert_eq!(deployment.configuration_key(&target, &options, &schedule).unwrap(), configuration);
            assert!(matches!(deployment.compile(&kernel, &target, &options, &schedule),
                Err(KernelCompilationError::Compiler(Error::Invalid { message }))
                    if message == "recorded compiler configuration cannot compile new kernels"));
        }
    }

    #[test]
    fn test_compiler_compile_native_output_limits() {
        if std::env::var("RYFT_RUN_TRITON_COMPILER_TESTS").ok().as_deref() != Some("1") {
            return;
        }
        let compiler = Compiler::new();
        let r#type = ArrayType::new_static(DataType::F32, [256]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let kernel = VerifiedKernel::new(&definition, 1024).unwrap();
        let target = Target::Cuda { major: 12, minor: 1 };
        let schedule = KernelSchedule::default();
        // One element per thread keeps this fixture's diagnostics focused on output validation.
        for (limit, expected) in [(1024, "compiler artifact exceeds its byte limit"), (8, "compiler")] {
            let options = Options::default().with_warp_count(8).unwrap().with_output_limits(1, limit).unwrap();
            let error = compiler.compile(&kernel, &target, &options, &schedule).unwrap_err();
            let KernelCompilationError::Compiler(Error::Compilation { message }) = error else {
                panic!("expected a native compilation error, got {error:?}");
            };
            assert_eq!(message, expected);
        }
    }

    #[test]
    fn test_copy_native_bytes() {
        let bytes = [1, 2, 3];
        assert_eq!(unsafe { copy_native_bytes(bytes.as_ptr(), bytes.len(), 3) }.unwrap(), bytes);
        assert_eq!(unsafe { copy_native_bytes(std::ptr::null(), 0, 3) }.unwrap(), Vec::<u8>::new());
        for (pointer, size, maximum) in [(std::ptr::null(), 1, 3), (bytes.as_ptr(), 3, 2)] {
            assert!(matches!(unsafe { copy_native_bytes(pointer, size, maximum) },
                Err(Error::Artifact { message }) if message == "native output is null or exceeds its size limit"));
        }
    }

    #[test]
    fn test_tokens() {
        assert_eq!(
            tokens(".entry /* comment */ kernel(\".entry fake\") // ignored\n{}").unwrap(),
            [".entry", "kernel", "(", ")", "{", "}"]
        );
        assert!(matches!(
            tokens("/* unterminated"),
            Err(Error::Artifact { message }) if message == "unterminated PTX comment"
        ));
        assert!(
            matches!(tokens("\"unterminated"), Err(Error::Artifact { message }) if message == "unterminated PTX string")
        );
    }

    #[test]
    fn test_validate_ptx() {
        let target = Target::Cuda { major: 12, minor: 1 };
        assert_eq!(validate_ptx(PTX, &target, &metadata()).unwrap(), "sm_121");
        for source in [
            PTX.replace("sm_121", "sm_80"),
            PTX.replace(".u64", ".u32"),
            PTX.replace("128", "256"),
            PTX.replace(".global", ".shared"),
            PTX.replace("9.0", "8.0"),
            format!("{PTX}\n{PTX}"),
        ] {
            assert!(matches!(validate_ptx(&source, &target, &metadata()), Err(Error::Artifact { message })
                if message == "PTX declarations differ from the native artifact contract"));
        }
        let mut changed = metadata();
        changed.argument_count = 2;
        assert!(matches!(validate_ptx(PTX, &target, &changed), Err(Error::Artifact { .. })));
    }
}
