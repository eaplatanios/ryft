//! Bounded native compiler invocation and concrete device artifact validation.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ryft_core::kernels::{
    KERNEL_CALL_OPERATION_NAME, KernelCompilationError, KernelCompiler, KernelSchedule, VerifiedKernel,
};
use ryft_core::{EffectClass, Typed};
use ryft_cuda::{
    CudaArtifactFormat, CudaKernelAbi, CudaKernelArtifact, CudaKernelLaunchDimensions, CudaKernelParameterType,
};
use ryft_rocm::{RocmKernelArtifact, RocmKernelLaunchDimensions};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::kernels::lowering;
use crate::kernels::{
    Artifact, COMPILER_SCHEMA_VERSION, CompiledKernel, Error, Options, TRITON_VERSION, Target, XLA_VERSION,
};

/// Explicit native compiler installation and optional caller-owned cancellation signal.
#[derive(Clone, Debug)]
pub struct Compiler {
    /// Absolute executable path; no implicit compiler selection occurs.
    executable: Option<PathBuf>,

    /// Recorded configuration permits AOT compatibility checks without a compiler installation.
    configuration: Option<Value>,

    /// Cancellation affects control flow, not semantic identity.
    cancellation: Arc<AtomicBool>,
}

/// Native products are decoded strictly before constructing a launchable artifact.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    /// Native protocol schema.
    schema: u32,

    /// Concrete platform.
    platform: String,

    /// Requested compiler architecture.
    requested_architecture: String,

    /// Actual exported function.
    entry_name: String,

    /// Verified physical pointer count after scratch argument removal.
    argument_count: usize,

    /// Actual warp count after native specialization.
    warp_count: u32,

    /// Requested warp count.
    requested_warp_count: u32,

    /// Requested pipeline stage count.
    stage_count: usize,

    /// Native threads in each warp or wavefront.
    threads_per_warp: u32,

    /// Actual thread block width.
    block_dimension_x: u32,

    /// Dynamic shared-memory allocation required at launch.
    shared_memory_bytes: u32,

    /// Unsupported external global scratch requirement.
    global_scratch_bytes: u64,

    /// Concrete image representation.
    artifact_format: String,
}

impl Compiler {
    /// Selects an absolute native executable path without starting a process or opening a GPU context.
    pub fn new(executable: PathBuf) -> Result<Self, Error> {
        if !executable.is_absolute() {
            return Err(Error::Invalid { message: "compiler executable must be an absolute path".into() });
        }
        Ok(Self { executable: Some(executable), configuration: None, cancellation: Arc::new(AtomicBool::new(false)) })
    }

    /// Restores validated compiler configuration for AOT loading without opening an executable or runtime.
    /// This compiler can check configuration compatibility but cannot compile new kernels. Configuration records
    /// are compatibility metadata, not authentication of native code or its producer.
    pub fn from_configuration(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() > 1024 * 1024 {
            return Err(Error::Invalid { message: "compiler configuration exceeds 1 MiB".into() });
        }
        let configuration: Value = serde_json::from_slice(bytes)?;
        let fields = configuration
            .as_object()
            .ok_or_else(|| Error::Invalid { message: "compiler configuration must be an object".into() })?;
        let expected = [
            "schema",
            "versions",
            "executable_sha256",
            "target",
            "warp_count",
            "pipeline_stages",
            "maximum_scratch_bytes",
        ];
        if fields.len() != expected.len()
            || expected.iter().any(|name| !fields.contains_key(*name))
            || configuration["schema"] != json!(COMPILER_SCHEMA_VERSION)
            || configuration["executable_sha256"].as_str().is_none_or(|hash| {
                hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            })
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
        Ok(Self {
            executable: None,
            configuration: Some(configuration),
            cancellation: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Returns the native executable, absent for recorded AOT configuration.
    pub fn executable(&self) -> Option<&Path> {
        self.executable.as_deref()
    }

    /// Uses a caller-owned signal to terminate the isolated native process group during compilation.
    pub fn with_cancellation(mut self, cancellation: Arc<AtomicBool>) -> Self {
        self.cancellation = cancellation;
        self
    }

    /// Computes actual executable and toolchain identity under the same bounds as compilation.
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
        let executable = self.executable.as_ref().unwrap();
        let mut file = File::open(executable).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Error::Unavailable { message: format!("compiler executable `{}` does not exist", executable.display()) }
            } else {
                Error::Io(error)
            }
        })?;
        if !file.metadata()?.is_file() {
            return Err(Error::Invalid { message: "compiler executable must be a regular file".into() });
        }
        let mut digest = Sha256::new();
        let mut buffer = [0u8; 65_536];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        let directory = tempfile::tempdir()?;
        let (stdout, _) = self.run(directory.path(), &["--version"], options)?;
        let versions: Value = serde_json::from_str(&stdout)?;
        Self::validate_versions(&versions, target)?;
        Ok(serde_json::to_vec(&json!({
            "schema": COMPILER_SCHEMA_VERSION,
            "versions": versions,
            "executable_sha256": format!("{:x}", digest.finalize()),
            "target": target,
            "warp_count": options.warp_count(),
            "pipeline_stages": stages,
            "maximum_scratch_bytes": schedule.maximum_scratch_bytes(),
        }))?)
    }

    /// Uses the same exact source and target-specific toolchain contract for live and recorded installations.
    fn validate_versions(versions: &Value, target: &Target) -> Result<(), Error> {
        let fixed = json!({
            "schema": 1, "protocol": 1, "xla": XLA_VERSION,
            "jax": "a7606f995e1a92707cbeb257e487fa53e7abe84b", "triton": TRITON_VERSION,
            "rocm_device_libs": "53996464fa8d94b182ac4aaa7dc3a109ab524f45",
        });
        let valid_toolchain = match target {
            Target::Cuda { .. } => {
                versions["cuda_toolkit_version"] == json!(13020) && versions["assembler_version"] == json!("13.0.88")
            }
            Target::Rocm { .. } => {
                versions["cuda_toolkit_version"]
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
        if !versions.as_object().is_some_and(|fields| fields.len() == 8)
            || fixed.as_object().unwrap().iter().any(|(name, value)| versions[name] != *value)
            || !valid_toolchain
        {
            return Err(Error::Invalid {
                message: "compiler installation differs from the qualified source and toolchain contract".into(),
            });
        }
        Ok(())
    }

    /// Rejects unsupported physical buffers and target-specific effects before starting a native process.
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

    /// Executes one native process group with bounded files, deadline and cancellation; always reaps the child.
    #[cfg(unix)]
    fn run(&self, directory: &Path, arguments: &[&str], options: &Options) -> Result<(String, String), Error> {
        use std::os::unix::process::CommandExt;

        if self.cancellation.load(Ordering::Acquire) {
            return Err(Error::Tool {
                reason: "cancelled before launch".into(),
                stdout: String::new(),
                stderr: String::new(),
            });
        }
        let stdout_path = directory.join("stdout.log");
        let stderr_path = directory.join("stderr.log");
        let executable = self.executable.as_ref().ok_or_else(|| Error::Unavailable {
            message: "recorded compiler configuration cannot compile new kernels".into(),
        })?;
        let mut child = Command::new(executable)
            .args(arguments)
            .current_dir(directory)
            .env_clear()
            .env("PATH", "/usr/local/cuda/bin:/usr/bin:/bin")
            .env("TMPDIR", directory)
            .stdin(Stdio::null())
            .stdout(File::create(&stdout_path)?)
            .stderr(File::create(&stderr_path)?)
            .process_group(0)
            .spawn()?;
        let start = Instant::now();
        let outcome = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Err(error) => break Err(format!("failed while observing child: {error}")),
                Ok(None) => {}
            }
            if self.cancellation.load(Ordering::Acquire) {
                break Err("cancelled".into());
            }
            if start.elapsed() >= options.process_timeout() {
                break Err("timed out".into());
            }
            let bounds = [
                (&stdout_path, options.maximum_diagnostic_bytes()),
                (&stderr_path, options.maximum_diagnostic_bytes()),
            ];
            if let Some(reason) = bounds.iter().find_map(|(path, limit)| match fs::metadata(path) {
                Ok(metadata) if metadata.len() > *limit as u64 => Some("exceeded diagnostic capture limit".to_owned()),
                Err(error) => Some(format!("failed to inspect diagnostic output: {error}")),
                _ => None,
            }) {
                break Err(reason);
            }
            if let Some(reason) =
                [("artifact", options.maximum_artifact_bytes()), ("metadata.json", options.maximum_diagnostic_bytes())]
                    .iter()
                    .find_map(|(name, limit)| match fs::metadata(directory.join(name)) {
                        Ok(metadata) if metadata.len() > *limit as u64 => {
                            Some("exceeded compiler product size limit".to_owned())
                        }
                        Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                            Some(format!("failed to inspect compiler product: {error}"))
                        }
                        _ => None,
                    })
            {
                break Err(reason);
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        unsafe extern "C" {
            fn kill(process: i32, signal: i32) -> i32;
        }
        // The negative child PID addresses only the fresh process group. Descendants must not survive completion,
        // failure or cancellation. Retain a cleanup error while still reaping the owned child below.
        let result = unsafe { kill(-(child.id() as i32), 9) };
        let cleanup = if result == 0 {
            None
        } else {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() == Some(3) { None } else { Some(error) }
        };
        child.wait()?;
        if let Some(error) = cleanup {
            return Err(error.into());
        }
        let read_diagnostic = |path: &Path| -> Result<String, Error> {
            let mut bytes = Vec::new();
            File::open(path)?.take(options.maximum_diagnostic_bytes() as u64).read_to_end(&mut bytes)?;
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        };
        let stdout = read_diagnostic(&stdout_path)?;
        let stderr = read_diagnostic(&stderr_path)?;
        if fs::metadata(&stdout_path)?.len().max(fs::metadata(&stderr_path)?.len())
            > options.maximum_diagnostic_bytes() as u64
        {
            return Err(Error::Tool { reason: "exceeded diagnostic capture limit".into(), stdout, stderr });
        }
        match outcome {
            Ok(status) if status.success() => Ok((stdout, stderr)),
            Ok(status) => Err(Error::Tool { reason: format!("exited with `{status}`"), stdout, stderr }),
            Err(reason) => Err(Error::Tool { reason, stdout, stderr }),
        }
    }

    /// Native compiler process isolation currently requires Unix process groups.
    #[cfg(not(unix))]
    fn run(&self, _directory: &Path, _arguments: &[&str], _options: &Options) -> Result<(String, String), Error> {
        Err(Error::Invalid { message: "native compiler process isolation requires Unix".into() })
    }

    /// Reads at most one byte beyond a validated limit, detecting files that grow after metadata inspection.
    fn read(path: &Path, limit: usize) -> Result<Vec<u8>, Error> {
        let mut bytes = Vec::new();
        File::open(path)?.take(limit as u64 + 1).read_to_end(&mut bytes)?;
        if bytes.len() > limit {
            return Err(Error::Artifact { message: "compiler output exceeds its size limit".into() });
        }
        Ok(bytes)
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
        lowering::lower(kernel, options, schedule).map_err(classify)?;
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
            let lowered = lowering::lower(kernel, options, schedule)?;
            if lowered.ttir.len() > 8 * 1024 * 1024 {
                return Err(Error::Invalid { message: "generated TTIR exceeds the native 8 MiB limit".into() });
            }
            let configuration_key = self.key(target, options, schedule)?;
            let (platform, architecture, threads_per_warp, format) = match target {
                Target::Cuda { major, minor } => ("cuda", format!("{major}.{minor}"), 32, "ptx"),
                Target::Rocm { architecture } => ("rocm", architecture.clone(), 64, "hsaco"),
            };
            let stages = schedule.pipeline_stages().map_or(2, |stages| stages.get());
            let directory = tempfile::tempdir()?;
            fs::write(directory.path().join("input.ttir"), &lowered.ttir)?;
            let (stdout, stderr) = self.run(
                directory.path(),
                &[
                    platform,
                    &architecture,
                    &options.warp_count().to_string(),
                    &stages.to_string(),
                    "input.ttir",
                    "artifact",
                    "metadata.json",
                ],
                options,
            )?;
            let metadata: Metadata = serde_json::from_slice(&Self::read(
                &directory.path().join("metadata.json"),
                options.maximum_diagnostic_bytes(),
            )?)?;
            if metadata.schema != 1
                || metadata.platform != platform
                || metadata.requested_architecture != architecture
                || metadata.entry_name != "ryft_kernel"
                || metadata.argument_count != lowered.parameter_types.len()
                || metadata.requested_warp_count != options.warp_count()
                || metadata.stage_count != stages
                || metadata.threads_per_warp != threads_per_warp
                || metadata.warp_count == 0
                || metadata.warp_count > 1024 / threads_per_warp
                || metadata.block_dimension_x != metadata.warp_count * threads_per_warp
                || metadata.global_scratch_bytes != 0
                || metadata.artifact_format != format
                || metadata.shared_memory_bytes > 1024 * 1024
                || schedule.maximum_scratch_bytes().is_some_and(|limit| metadata.shared_memory_bytes as usize > limit)
            {
                return Err(Error::Artifact {
                    message: "native metadata differs from the requested entry, target or resource contract".into(),
                });
            }
            let bytes = Self::read(&directory.path().join("artifact"), options.maximum_artifact_bytes())?;
            let artifact = match target {
                Target::Cuda { .. } => {
                    let ptx = std::str::from_utf8(&bytes)
                        .map_err(|_| Error::Artifact { message: "PTX is not UTF-8".into() })?;
                    let target_architecture = validate_ptx(ptx, target, &metadata)?;
                    Artifact::Cuda(CudaKernelArtifact::new(
                        CudaArtifactFormat::Ptx,
                        bytes,
                        "ryft_kernel",
                        target_architecture,
                        CudaKernelLaunchDimensions::new(
                            lowered.grid,
                            [metadata.block_dimension_x, 1, 1],
                            metadata.shared_memory_bytes,
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
                        metadata.shared_memory_bytes,
                    )?,
                )?),
            };
            if self.key(target, options, schedule)? != configuration_key {
                return Err(Error::Invalid { message: "compiler installation changed during compilation".into() });
            }
            Ok(CompiledKernel {
                artifact,
                parameter_types: lowered.parameter_types,
                semantic_key: kernel.definition().semantic_key()?,
                configuration_key,
                diagnostics: format!("stdout:\n{stdout}\nstderr:\n{stderr}"),
            })
        };
        compile().map_err(classify)
    }
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
        Metadata {
            schema: 1,
            platform: "cuda".into(),
            requested_architecture: "12.1".into(),
            entry_name: "ryft_kernel".into(),
            argument_count: 1,
            warp_count: 4,
            requested_warp_count: 4,
            stage_count: 2,
            threads_per_warp: 32,
            block_dimension_x: 128,
            shared_memory_bytes: 0,
            global_scratch_bytes: 0,
            artifact_format: "ptx".into(),
        }
    }

    /// A portable definition used to exercise the complete compile protocol with controlled native products.
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
            "schema": 1, "protocol": 1, "xla": XLA_VERSION,
            "jax": "a7606f995e1a92707cbeb257e487fa53e7abe84b", "triton": TRITON_VERSION,
            "rocm_device_libs": "53996464fa8d94b182ac4aaa7dc3a109ab524f45",
            "cuda_toolkit_version": 13020, "assembler_version": "13.0.88",
        })
    }

    /// Creates a controlled native installation whose products are ordinary sibling files.
    #[cfg(unix)]
    fn installation(directory: &Path, artifact: &[u8], metadata: &Value, changed: bool) -> Compiler {
        use std::os::unix::fs::PermissionsExt;

        fs::write(directory.join("versions.json"), serde_json::to_vec(&versions()).unwrap()).unwrap();
        fs::write(directory.join("product"), artifact).unwrap();
        fs::write(directory.join("product.json"), serde_json::to_vec(metadata).unwrap()).unwrap();
        let executable = directory.join("compiler");
        let script = indoc! {r#"
            #!/bin/sh
            directory=$(dirname "$0")
            if [ "$1" = "--version" ]; then
                cat "$directory/versions.json"
            else
                cp "$directory/product" "$6"
                cp "$directory/product.json" "$7"
            fi
        "#};
        fs::write(
            &executable,
            if changed { format!("{script}printf '# changed\\n' >> \"$0\"\n") } else { script.into() },
        )
        .unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
        Compiler::new(executable).unwrap()
    }

    #[test]
    fn test_compiler_new() {
        assert!(Compiler::new(PathBuf::from("/compiler")).is_ok());
        assert!(matches!(Compiler::new(PathBuf::from("compiler")), Err(Error::Invalid { message })
            if message == "compiler executable must be an absolute path"));
    }

    #[test]
    fn test_compiler_from_configuration() {
        let configuration = json!({
            "schema": COMPILER_SCHEMA_VERSION, "versions": versions(), "executable_sha256": "a".repeat(64),
            "target": Target::Cuda { major: 12, minor: 1 }, "warp_count": 4,
            "pipeline_stages": 2, "maximum_scratch_bytes": null,
        });
        let bytes = serde_json::to_vec(&configuration).unwrap();
        let restored = Compiler::from_configuration(&bytes).unwrap();
        assert_eq!(restored.executable(), None);
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
    fn test_compiler_executable() {
        assert_eq!(Compiler::new("/compiler".into()).unwrap().executable(), Some(Path::new("/compiler")));
    }

    #[test]
    fn test_compiler_with_cancellation() {
        let cancellation = Arc::new(AtomicBool::new(false));
        let compiler = Compiler::new("/compiler".into()).unwrap().with_cancellation(cancellation.clone());
        cancellation.store(true, Ordering::Release);
        assert!(compiler.cancellation.load(Ordering::Acquire));
    }

    #[cfg(unix)]
    #[test]
    fn test_compiler_key() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("compiler");
        let versions = versions();
        let script = format!("#!/bin/sh\nprintf '%s\\n' '{versions}'\n");
        fs::write(&path, &script).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        let compiler = Compiler::new(path.clone()).unwrap();
        let target = Target::Cuda { major: 12, minor: 1 };
        let schedule = KernelSchedule::default();
        let key = compiler.key(&target, &Options::default(), &schedule).unwrap();
        assert_eq!(
            compiler
                .key(&target, &Options::default().with_process_timeout(Duration::from_secs(1)).unwrap(), &schedule)
                .unwrap(),
            key
        );
        assert_ne!(compiler.key(&target, &Options::default().with_warp_count(8).unwrap(), &schedule).unwrap(), key);
        let recorded = Compiler::from_configuration(&key).unwrap();
        assert_eq!(recorded.executable(), None);
        assert_eq!(recorded.key(&target, &Options::default(), &schedule).unwrap(), key);
        assert!(matches!(
            recorded.key(&target, &Options::default().with_warp_count(8).unwrap(), &schedule),
            Err(Error::Invalid { message })
                if message == "requested options differ from recorded compiler configuration"
        ));
        fs::write(&path, format!("{script}# changed executable\n")).unwrap();
        assert_ne!(compiler.key(&target, &Options::default(), &schedule).unwrap(), key);
        fs::write(&path, script.replace("13.0.88", "13.1.0")).unwrap();
        assert!(matches!(compiler.key(&target, &Options::default(), &schedule), Err(Error::Invalid { message })
            if message == "compiler installation differs from the qualified source and toolchain contract"));
    }

    #[cfg(unix)]
    #[test]
    fn test_compiler_run() {
        let compiler = Compiler::new("/bin/sh".into()).unwrap();
        let directory = tempfile::tempdir().unwrap();
        assert_eq!(
            compiler
                .run(directory.path(), &["-c", "printf output; printf diagnostic >&2"], &Options::default())
                .unwrap(),
            ("output".into(), "diagnostic".into())
        );
        assert!(matches!(compiler.run(directory.path(), &["-c", "printf partial; exit 7"], &Options::default()),
            Err(Error::Tool { stdout, stderr, .. }) if stdout == "partial" && stderr.is_empty()));
        let options = Options::default().with_output_limits(32, 4).unwrap();
        assert!(matches!(compiler.run(directory.path(), &["-c", "printf 123456"], &options),
            Err(Error::Tool { reason, stdout, stderr }) if reason == "exceeded diagnostic capture limit"
                && stdout == "1234" && stderr.is_empty()));
    }

    #[cfg(unix)]
    #[test]
    fn test_compiler_run_cancellation_and_timeout() {
        let directory = tempfile::tempdir().unwrap();
        let cancellation = Arc::new(AtomicBool::new(true));
        let compiler = Compiler::new("/bin/sh".into()).unwrap().with_cancellation(cancellation.clone());
        assert!(matches!(compiler.run(directory.path(), &["-c", "exit 0"], &Options::default()),
            Err(Error::Tool { reason, stdout, stderr })
                if reason == "cancelled before launch" && stdout.is_empty() && stderr.is_empty()));
        cancellation.store(false, Ordering::Release);
        let options = Options::default().with_process_timeout(Duration::from_millis(20)).unwrap();
        assert!(matches!(compiler.run(directory.path(), &["-c", "sleep 2"], &options),
            Err(Error::Tool { reason, .. }) if reason == "timed out"));
    }

    #[test]
    fn test_compiler_read() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("output");
        fs::write(&path, [1, 2, 3]).unwrap();
        assert_eq!(Compiler::read(&path, 3).unwrap(), [1, 2, 3]);
        assert!(matches!(Compiler::read(&path, 2), Err(Error::Artifact { message })
            if message == "compiler output exceeds its size limit"));
    }

    #[test]
    fn test_compiler_admit() {
        let directory = tempfile::tempdir().unwrap();
        let compiler = Compiler::new(directory.path().join("missing")).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [32, 32]);
        let definition = matrix::definition(&r#type, &r#type).unwrap();
        let verified = VerifiedKernel::new(&definition, 1024).unwrap();
        let options = Options::default();
        let schedule = KernelSchedule::default();
        assert!(matches!(
            compiler.admit(&verified, &Target::Rocm { architecture: "gfx942".into() }, &options, &schedule),
            Err(KernelCompilationError::Unsupported { operation: "ordered_assertion", .. })
        ));
        let r#type = ArrayType::new_static(DataType::F32, [256]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let verified = VerifiedKernel::new(&definition, 1024).unwrap();
        assert!(matches!(compiler.admit(&verified, &Target::Cuda { major: 12, minor: 1 }, &options, &schedule),
            Err(KernelCompilationError::Unavailable { message }) if message.contains("does not exist")));
    }

    #[test]
    fn test_compiler_admit_and_compile_empty_parameters() {
        let directory = tempfile::tempdir().unwrap();
        let compiler = Compiler::new(directory.path().join("missing")).unwrap();
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
        let directory = tempfile::tempdir().unwrap();
        let compiler = Compiler::new(directory.path().join("missing")).unwrap();
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
                assert!(matches!(
                    compiler.admit(&verified, &target, &options, &schedule),
                    Err(KernelCompilationError::Unavailable { .. })
                ));
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

    #[cfg(unix)]
    #[test]
    fn test_compiler_compile() {
        let r#type = ArrayType::new_static(DataType::F32, [256]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let verified = VerifiedKernel::new(&definition, 1024).unwrap();
        let target = Target::Cuda { major: 12, minor: 1 };
        let options = Options::default();
        let schedule = KernelSchedule::default();
        let metadata = json!({
            "schema": 1, "platform": "cuda", "requested_architecture": "12.1", "entry_name": "ryft_kernel",
            "argument_count": 3, "warp_count": 4, "requested_warp_count": 4, "stage_count": 2,
            "threads_per_warp": 32, "block_dimension_x": 128, "shared_memory_bytes": 0,
            "global_scratch_bytes": 0, "artifact_format": "ptx",
        });
        let ptx = PTX.replace(
            ".param .u64 .ptr .global .align 1 pointer",
            ".param .u64 .ptr .global .align 1 first, \
             .param .u64 .ptr .global .align 1 second, \
             .param .u64 .ptr .global .align 1 output",
        );
        let directory = tempfile::tempdir().unwrap();
        let compiler = installation(directory.path(), ptx.as_bytes(), &metadata, false);
        compiler.admit(&verified, &target, &options, &schedule).unwrap();
        let compiled = compiler.compile(&verified, &target, &options, &schedule).unwrap();
        assert_eq!(compiled.parameter_types(), &[r#type.clone(), r#type.clone(), r#type]);
        assert_eq!(compiled.semantic_key(), definition.semantic_key().unwrap());
        assert!(matches!(compiled.artifact(), Artifact::Cuda(_)));
        for (field, value) in [
            ("argument_count", json!(4)),
            ("platform", json!("rocm")),
            ("block_dimension_x", json!(256)),
            ("global_scratch_bytes", json!(1)),
            ("unexpected", json!(0)),
        ] {
            let mut changed = metadata.clone();
            changed[field] = value;
            fs::write(directory.path().join("product.json"), serde_json::to_vec(&changed).unwrap()).unwrap();
            assert!(matches!(
                compiler.compile(&verified, &target, &options, &schedule),
                Err(KernelCompilationError::Compiler(Error::Artifact { .. } | Error::Json(_)))
            ));
        }
        fs::write(directory.path().join("product.json"), serde_json::to_vec(&metadata).unwrap()).unwrap();
        fs::write(directory.path().join("product"), ptx.replace("sm_121", "sm_80")).unwrap();
        assert!(matches!(
            compiler.compile(&verified, &target, &options, &schedule),
            Err(KernelCompilationError::Compiler(Error::Artifact { .. }))
        ));
        fs::write(directory.path().join("product"), [0xff]).unwrap();
        assert!(matches!(compiler.compile(&verified, &target, &options, &schedule),
            Err(KernelCompilationError::Compiler(Error::Artifact { message })) if message == "PTX is not UTF-8"));
        let mut rocm_metadata = metadata.clone();
        rocm_metadata["platform"] = json!("rocm");
        rocm_metadata["requested_architecture"] = json!("gfx942");
        rocm_metadata["threads_per_warp"] = json!(64);
        rocm_metadata["block_dimension_x"] = json!(256);
        rocm_metadata["artifact_format"] = json!("hsaco");
        fs::write(directory.path().join("product.json"), serde_json::to_vec(&rocm_metadata).unwrap()).unwrap();
        assert!(matches!(
            compiler.compile(&verified, &Target::Rocm { architecture: "gfx942".into() }, &options, &schedule),
            Err(KernelCompilationError::Compiler(Error::Rocm(_)))
        ));
        let changed = installation(directory.path(), ptx.as_bytes(), &metadata, true);
        assert!(matches!(changed.compile(&verified, &target, &options, &schedule),
            Err(KernelCompilationError::Compiler(Error::Invalid { message }))
                if message == "compiler installation changed during compilation"));
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
