use std::fmt::Display;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

use anyhow::{Context, Result, anyhow, bail};
use flate2::Compression;
use flate2::write::GzEncoder;
use reqwest::Url;
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use zip::ZipArchive;

/// Name of the environment variable that contains the path to a precompiled `ryft-xla-sys` archive.
static RYFT_XLA_SYS_ARCHIVE: &str = "RYFT_XLA_SYS_ARCHIVE";

/// Name of the environment variable that contains the path to a precompiled PJRT CUDA 12 plugin.
static PJRT_PLUGIN_CUDA_12_LIB: &str = "PJRT_PLUGIN_CUDA_12_LIB";

/// Name of the environment variable that contains the path to a precompiled PJRT CUDA 13 plugin.
static PJRT_PLUGIN_CUDA_13_LIB: &str = "PJRT_PLUGIN_CUDA_13_LIB";

/// Name of the environment variable that contains the path to a precompiled PJRT ROCm 7 plugin.
static PJRT_PLUGIN_ROCM_7_LIB: &str = "PJRT_PLUGIN_ROCM_7_LIB";

/// Name of the environment variable that contains the path to a precompiled PJRT TPU plugin.
static PJRT_PLUGIN_TPU_LIB: &str = "PJRT_PLUGIN_TPU_LIB";

/// Name of the environment variable that contains the path to a precompiled PJRT Neuron plugin.
static PJRT_PLUGIN_NEURON_LIB: &str = "PJRT_PLUGIN_NEURON_LIB";

/// Name of the environment variable that contains the path to a precompiled PJRT Metal plugin.
static PJRT_PLUGIN_METAL_LIB: &str = "PJRT_PLUGIN_METAL_LIB";

/// URL paired with an expected SHA-256 checksum for verifying downloads.
#[derive(Clone, PartialEq, Eq, Hash)]
struct UrlWithChecksum {
    /// Remote URL.
    url: Url,

    /// Hex-encoded SHA-256 checksum for the content at `url`.
    checksum: String,
}

impl UrlWithChecksum {
    /// Downloads the contents pointed to by the current [`UrlWithChecksum::url`] to `path`, verifying that they have
    /// the expected checksum (and re-downloading if there is a mismatch).
    fn download(&self, path: &Path) -> Result<()> {
        println!("cargo::warning=Attempting to download file from '{}'.", self.url);

        // Check if the file already exists and that the checksum matches.
        if path.exists() {
            println!("cargo::warning=Found cached file at '{}', verifying checksum...", path.display());
            if self.verify_checksum(&path).is_ok() {
                println!("cargo::warning=Checksum verification succeeded; skipping download.");
                return Ok(());
            } else {
                println!("cargo::warning=Checksum verification failed; re-downloading file...");
                fs::remove_file(&path)?;
            }
        }

        // Download the file content.
        let client = Client::new();
        let response = client.get(self.url.clone()).send()?;
        if !response.status().is_success() {
            bail!("encountered HTTP error ({}) while downloading '{}'", response.status(), self.url);
        }

        // Create the parent directory of the destination [`Path`], if it does not exist already.
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Write the downloaded content into a [`File`] at the destination [`Path`].
        let content = response.bytes()?;
        let mut file = File::create(path)?;
        file.write_all(&content)?;

        // Verify the checksum of the downloaded file.
        self.verify_checksum(path)?;

        println!("cargo::warning=Successfully downloaded '{}' from '{}'.", path.display(), self.url);
        Ok(())
    }

    /// Verifies that the file at `path` matches the current [`UrlWithChecksum::checksum`].
    fn verify_checksum(&self, path: &Path) -> Result<()> {
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        let hash_bytes = hasher.finalize();
        let hash = hex::encode(hash_bytes);
        if hash != self.checksum {
            bail!("checksum mismatch (expected '{}' but got '{}')", self.checksum, hash);
        }
        Ok(())
    }
}

/// Target operating system for this build that is determined based on the `CARGO_CFG_TARGET_OS` environment variable.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum OperatingSystem {
    Linux,
    MacOS,
    Windows,
}

impl OperatingSystem {
    /// Parses the target [`OperatingSystem`] of this build from the `CARGO_CFG_TARGET_OS` environment variable.
    fn from_environment() -> Result<Self> {
        match env::var("CARGO_CFG_TARGET_OS").map_err(|_| anyhow!("failed to get the target OS"))?.as_str() {
            "linux" => Ok(Self::Linux),
            "macos" => Ok(Self::MacOS),
            "windows" => Ok(Self::Windows),
            os => Err(anyhow!("unsupported operating system: '{os}'")),
        }
    }

    /// Returns the filename prefix for shared/dynamic or static libraries on this [`OperatingSystem`].
    fn library_prefix(&self) -> &'static str {
        match self {
            OperatingSystem::Linux => "lib",
            OperatingSystem::MacOS => "lib",
            OperatingSystem::Windows => "",
        }
    }

    /// Returns the file extension used for shared/dynamic libraries on this [`OperatingSystem`].
    fn dynamic_library_extension(&self) -> &'static str {
        match self {
            OperatingSystem::Linux => "so",
            OperatingSystem::MacOS => "dylib",
            OperatingSystem::Windows => "dll",
        }
    }

    /// Returns the file extension used for static libraries on this [`OperatingSystem`].
    fn static_library_extension(&self) -> &'static str {
        match self {
            OperatingSystem::Linux => "a",
            OperatingSystem::MacOS => "a",
            OperatingSystem::Windows => "lib",
        }
    }
}

impl Display for OperatingSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperatingSystem::Linux => write!(f, "linux"),
            OperatingSystem::MacOS => write!(f, "macos"),
            OperatingSystem::Windows => write!(f, "windows"),
        }
    }
}

/// Target CPU architecture for this build that is determined based on the `CARGO_CFG_TARGET_ARCH` environment variable.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Architecture {
    /// x86_64 / amd64 architecture.
    X86_64,

    /// AArch64 / arm64 architecture.
    AArch64,
}

impl Architecture {
    /// Parses the target [`Architecture`] of this build from the `CARGO_CFG_TARGET_ARCH` environment variable.
    fn from_environment() -> Result<Self> {
        match env::var("CARGO_CFG_TARGET_ARCH")
            .map_err(|_| anyhow!("failed to get the target architecture"))?
            .as_str()
        {
            "x86_64" => Ok(Self::X86_64),
            "aarch64" => Ok(Self::AArch64),
            architecture => Err(anyhow!("unsupported architecture: '{architecture}'")),
        }
    }
}

impl Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Architecture::X86_64 => write!(f, "amd64"),
            Architecture::AArch64 => write!(f, "arm64"),
        }
    }
}

/// Target accelerator device/backend variant for which to build or fetch the native XLA artifacts.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Device {
    /// CPU (i.e., no accelerator).
    Cpu,

    /// Nvidia CUDA 12 devices/GPUs.
    Cuda12,

    /// Nvidia CUDA 13 devices/GPUs.
    Cuda13,

    /// AMD ROCm 7 devices/GPUs.
    Rocm7,

    /// Google TPU devices.
    Tpu,

    /// AWS Neuron devices (i.e., Inferentia and Trainium).
    Neuron,

    /// Apple Silicon devices using Metal.
    Metal,
}

impl Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda12 => write!(f, "cuda-12"),
            Device::Cuda13 => write!(f, "cuda-13"),
            Device::Rocm7 => write!(f, "rocm-7"),
            Device::Tpu => write!(f, "tpu"),
            Device::Neuron => write!(f, "neuron"),
            Device::Metal => write!(f, "metal"),
        }
    }
}

/// Type of XLA artifact that needs to be downloaded or built.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Artifact {
    /// Core XLA bindings library for Ryft.
    RyftXlaSys,

    /// PJRT plugin.
    PjrtPlugin,
}

impl Artifact {
    /// Returns the name of this [`Artifact`], which is used for logging purposes.
    fn name(&self) -> &str {
        match self {
            Artifact::RyftXlaSys => "ryft-xla-sys",
            Artifact::PjrtPlugin => "pjrt-plugin",
        }
    }
}

/// Build configuration that encompasses information about the target of this build.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct BuildConfiguration {
    /// Target [`OperatingSystem`] of this build.
    operating_system: OperatingSystem,

    /// Target [`Architecture`] of this build.
    architecture: Architecture,

    /// Target [`Device`] of this build.
    device: Device,
}

impl BuildConfiguration {
    /// Parses the target [`BuildConfiguration`] of this build from the `CARGO_CFG_TARGET_OS`
    /// and `CARGO_CFG_TARGET_ARCH` environment variables.
    fn from_environment() -> Result<Self> {
        Ok(Self {
            operating_system: OperatingSystem::from_environment()?,
            architecture: Architecture::from_environment()?,
            device: Device::Cpu,
        })
    }

    /// Downloads or builds the `ryft-xla-sys` native library and emits linking directives for it.
    fn configure_ryft_xla_sys(&self) {
        let ryft_xla_sys_directory = self.artifact_directory(Artifact::RyftXlaSys).unwrap();
        let library_directory = ryft_xla_sys_directory.join("lib");
        let library_directory = library_directory.canonicalize().unwrap_or(library_directory);

        // Configure static linking for the native library.
        println!("cargo::rustc-link-lib=static=ryft-xla-sys");
        match &self.operating_system {
            OperatingSystem::Linux => {
                println!("cargo::rustc-link-lib=stdc++");
                println!("cargo::rustc-link-arg=-g");
                println!("cargo::rustc-link-search=native={}", library_directory.display());
                println!("cargo::rustc-link-arg=-Wl,-rpath,{}", library_directory.display());
            }
            OperatingSystem::MacOS => {
                println!("cargo::rustc-link-lib=c++");
                println!("cargo::rustc-link-lib=framework=CoreFoundation");
                println!("cargo::rustc-link-arg=-g");
                println!("cargo::rustc-link-search=native={}", library_directory.display());
                println!("cargo::rustc-link-arg=-Wl,-rpath,{}", library_directory.display());
            }
            OperatingSystem::Windows => {
                println!("cargo::rustc-link-arg=/DEBUG");
                println!("cargo::rustc-link-search=native={}", library_directory.display());
                println!("cargo::rustc-env=RUSTFLAGS=-C target-feature=+crt-static");
            }
        }

        #[cfg(feature = "generate-bindings")]
        self.generate_c_bindings(&ryft_xla_sys_directory);
    }

    /// Generates the C API bindings for the `ryft-xla-sys` library using [`bindgen`].
    #[cfg(feature = "generate-bindings")]
    fn generate_c_bindings(&self, ryft_xla_sys_directory: &Path) {
        let out_dir = env::var("OUT_DIR").expect("the `OUT_DIR` environment variable is not set");
        let include_path = ryft_xla_sys_directory.join("include");
        let xla_include_path = include_path.join("xla");
        let pjrt_include_path = xla_include_path.join("pjrt").join("c");
        let mlir_c_include_path = include_path.join("mlir-c");
        let stablehlo_integrations_include_path = include_path.join("stablehlo").join("integrations");
        bindgen::builder()
            .header(pjrt_include_path.join("pjrt_c_api.h").to_str().unwrap())
            .header(pjrt_include_path.join("pjrt_c_api_ffi_extension.h").to_str().unwrap())
            .header(pjrt_include_path.join("pjrt_c_api_layouts_extension.h").to_str().unwrap())
            .header(pjrt_include_path.join("pjrt_c_api_memory_descriptions_extension.h").to_str().unwrap())
            .header(pjrt_include_path.join("pjrt_c_api_phase_compile_extension.h").to_str().unwrap())
            .header(pjrt_include_path.join("pjrt_c_api_profiler_extension.h").to_str().unwrap())
            .header(xla_include_path.join("ffi").join("api").join("c_api.h").to_str().unwrap())
            .header(xla_include_path.join("service").join("custom_call_status.h").to_str().unwrap())
            .header(
                xla_include_path
                    .join("service")
                    .join("spmd")
                    .join("shardy")
                    .join("integrations")
                    .join("c")
                    .join("passes.h")
                    .to_str()
                    .unwrap(),
            )
            .header(xla_include_path.join("mlir_hlo").join("bindings").join("c").join("Attributes.h").to_str().unwrap())
            .header(xla_include_path.join("mlir_hlo").join("bindings").join("c").join("Dialects.h").to_str().unwrap())
            .header(xla_include_path.join("mlir_hlo").join("bindings").join("c").join("Passes.h").to_str().unwrap())
            .header(xla_include_path.join("mlir_hlo").join("bindings").join("c").join("Types.h").to_str().unwrap())
            .header(stablehlo_integrations_include_path.join("c").join("ChloAttributes.h").to_str().unwrap())
            .header(stablehlo_integrations_include_path.join("c").join("ChloDialect.h").to_str().unwrap())
            .header(stablehlo_integrations_include_path.join("c").join("InterpreterDialect.h").to_str().unwrap())
            .header(stablehlo_integrations_include_path.join("c").join("StablehloAttributes.h").to_str().unwrap())
            .header(stablehlo_integrations_include_path.join("c").join("StablehloDialect.h").to_str().unwrap())
            .header(stablehlo_integrations_include_path.join("c").join("StablehloDialectApi.h").to_str().unwrap())
            .header(stablehlo_integrations_include_path.join("c").join("StablehloPasses.h").to_str().unwrap())
            .header(stablehlo_integrations_include_path.join("c").join("StablehloTypes.h").to_str().unwrap())
            .header(stablehlo_integrations_include_path.join("c").join("StablehloUnifiedApi.h").to_str().unwrap())
            .header(stablehlo_integrations_include_path.join("c").join("VhloDialect.h").to_str().unwrap())
            .header(include_path.join("shardy").join("integrations").join("c").join("attributes.h").to_str().unwrap())
            .header(include_path.join("shardy").join("integrations").join("c").join("dialect.h").to_str().unwrap())
            .header(include_path.join("shardy").join("integrations").join("c").join("passes.h").to_str().unwrap())
            .header(mlir_c_include_path.join("AffineExpr.h").to_str().unwrap())
            .header(mlir_c_include_path.join("AffineMap.h").to_str().unwrap())
            .header(mlir_c_include_path.join("BuiltinAttributes.h").to_str().unwrap())
            .header(mlir_c_include_path.join("BuiltinTypes.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Conversion.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Debug.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Diagnostics.h").to_str().unwrap())
            .header(mlir_c_include_path.join("ExecutionEngine.h").to_str().unwrap())
            .header(mlir_c_include_path.join("IR.h").to_str().unwrap())
            .header(mlir_c_include_path.join("IntegerSet.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Interfaces.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Pass.h").to_str().unwrap())
            .header(mlir_c_include_path.join("RegisterEverything.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Rewrite.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Support.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Transforms.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("AMDGPU.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("Arith.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("Async.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("ControlFlow.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("EmitC.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("Func.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("GPU.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("Index.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("IRDL.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("Linalg.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("LLVM.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("Math.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("MemRef.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("MLProgram.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("NVGPU.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("NVVM.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("PDL.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("Quant.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("ROCDL.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("SCF.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("Shape.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("SMT.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("SparseTensor.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("Tensor.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("Transform.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("Transform").join("Interpreter.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Dialect").join("Vector.h").to_str().unwrap())
            .header(mlir_c_include_path.join("Target").join("LLVMIR.h").to_str().unwrap())
            .clang_arg(format!("-I{}", include_path.display()))
            .allowlist_item("GetPjrtApi")
            .allowlist_item("XLA_FFI.*")
            .allowlist_item("XlaCustomCallStatus.*")
            .allowlist_item("PJRT.*")
            .allowlist_item("Mlir.*")
            .allowlist_item("mlir.*")
            .allowlist_item("chlo.*")
            .allowlist_item("stablehlo.*")
            .allowlist_item("sdy.*")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .unwrap()
            .write_to_file(Path::new(&out_dir).join("bindings.rs"))
            .expect("failed to generate C bindings");
    }

    /// Downloads or builds a PJRT plugin for the provided [`Device`] and sets a build-time environment variable
    /// to the path of the resulting shared library such that it can be loaded by `ryft-pjrt`.
    fn configure_pjrt_plugin(&self, device: Device) {
        let build_configuration = Self { device, ..self.clone() };
        let plugin_directory = build_configuration.artifact_directory(Artifact::PjrtPlugin).unwrap();
        let plugin_path = plugin_directory.join(build_configuration.pjrt_plugin_library_file_name());
        println!(
            "cargo:rustc-env=RYFT_PJRT_PLUGIN_{}={}",
            device.to_string().to_uppercase().replace("-", "_"),
            plugin_path.display(),
        );
    }

    /// Returns the expected PJRT plugin library file name for this [`BuildConfiguration`].
    fn pjrt_plugin_library_file_name(&self) -> String {
        format!(
            "{}pjrt-plugin-{}.{}",
            self.operating_system.library_prefix(),
            self.device,
            self.operating_system.dynamic_library_extension(),
        )
    }

    /// Returns a directory containing the requested [`Artifact`] by attempting to find it using the following
    /// approaches, in this order, and returning an [`Err`] if it is not found:
    ///
    ///   1. If the extracted archive containing the artifact is found in a local cache (e.g., from an earlier build),
    ///      return it. Otherwise, move on to step 2. Note that steps 2 and 3, if successful, will update the local
    ///      cache so that the extracted [`Artifact`] can be reused in subsequent builds.
    ///   2. If a path is provided via the appropriate environment variable (e.g., `RYFT_XLA_SYS_ARCHIVE` or
    ///      `PJRT_PLUGIN_*_LIB`), extract the archive pointed to by that path, if it is a `.tar.gz` or `.whl` archive,
    ///      and return the extracted path. Note that the archive will be extracted into a local cache directory
    ///      so that it can be reused in subsequent builds.
    ///   3. Otherwise, try to download and extract a precompiled archive and return the extracted path if successful.
    ///      Note that the downloaded archive will be extracted into a local cache directory so that it can be reused
    ///      in subsequent builds.
    ///   4. Otherwise, try to build the artifact from source using Bazel. Note that this is not supported when using
    ///      `cargo vendor` because it requires having access to the internet during the build process.
    ///
    /// When the resolved path is a `.tar.gz` or `.whl` archive, it is extracted into the cache directory and
    /// any existing extracted directory is replaced to keep the contents up to date. For extracted
    /// `ryft-xla-sys` archives, post-processing (e.g., renaming the static library) is applied.
    fn artifact_directory(&self, artifact: Artifact) -> Result<PathBuf> {
        let archive_name = self.precompiled_artifact_name(artifact);
        let extracted_path = dirs::cache_dir()
            .map(|cache_dir| cache_dir.join("ryft").join("xla"))
            .unwrap_or(Path::new(&env::var("OUT_DIR").with_context(|| "`OUT_DIR` not set")?).join("cache"))
            .join("extracted")
            .join(archive_name.trim_end_matches(".tar.gz").trim_end_matches(".whl"));

        // Check if the extracted artifact already exists in the cache and return immediately if it does.
        let cached_artifact_found = match artifact {
            Artifact::RyftXlaSys => fs::exists(extracted_path.join("lib").join(format!(
                "{}ryft-xla-sys.{}",
                self.operating_system.library_prefix(),
                self.operating_system.static_library_extension()
            )))
            .unwrap_or(false),
            Artifact::PjrtPlugin => {
                fs::exists(extracted_path.join(self.pjrt_plugin_library_file_name())).unwrap_or(false)
            }
        };
        if cached_artifact_found {
            return Ok(extracted_path);
        }

        // Check if a path to the archive has already been provided via the corresponding environment variable.
        let artifact_path = match self.artifact_path_from_environment(artifact) {
            Ok(artifact_path) => artifact_path,
            Err(error) => {
                println!("cargo::warning={error}\nAttempting to download a precompiled artifact instead.");

                // Try to download a precompiled build artifact if one exists.
                match self.download_precompiled_artifact_archive(artifact) {
                    Ok(artifact_path) => artifact_path,
                    Err(error) => {
                        println!("cargo::warning={error}\nAttempting to build from source instead.");

                        // Try to build the artifact from source using Bazel.
                        match self.build_artifact(artifact) {
                            Ok(artifact_path) => artifact_path,
                            Err(error) => bail!(error),
                        }
                    }
                }
            }
        };

        // Check if the artifact path points to a `.tar.gz` or a `.whl` file that needs extraction
        // and extract it if it does.
        let extension = artifact_path.extension().and_then(|extension| extension.to_str());

        if !matches!(extension, Some("gz" | "whl")) {
            Ok(artifact_path)
        } else {
            // Construct a path in the user's cache directory in which to extract the artifact archive, if needed.
            let cache_dir = dirs::cache_dir()
                .map(|cache_dir| cache_dir.join("ryft").join("xla"))
                .unwrap_or(Path::new(&env::var("OUT_DIR").with_context(|| "`OUT_DIR` not set")?).join("cache"));
            let extracted_path = cache_dir.join("extracted").join(
                artifact_path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .trim_end_matches(".tar.gz")
                    .trim_end_matches(".whl"),
            );

            // If the extracted path already exists, delete it and re-extract the archive.
            // This is done to ensure that the extracted files are always up to date.
            if extracted_path.exists() {
                fs::remove_dir_all(&extracted_path)
                    .with_context(|| format!("failed to delete the {} path", extracted_path.display()))?;
            }

            // Create a directory for the extracted files.
            fs::create_dir_all(&extracted_path)?;

            // Extract the artifact archive.
            if extension == Some("gz") {
                let tar_gz = File::open(artifact_path)?;
                let tar = flate2::read::GzDecoder::new(tar_gz);
                let mut archive = tar::Archive::new(tar);
                archive.unpack(&extracted_path)?;
            } else if extension == Some("whl") {
                let archive_file = File::open(artifact_path)?;
                let mut archive = ZipArchive::new(archive_file)?;
                archive.extract(&extracted_path)?;
            }

            // Make any file renames that are necessary for downstream code to function as expected.
            match artifact {
                Artifact::RyftXlaSys => {
                    // Rename the `ryft-xla-sys` static library to remove the `-static` suffix from its name.
                    // This makes it such that the `links = "ryft-xla-sys"` attribute in the `Cargo.toml` is accurate.
                    let file_name_prefix = self.operating_system.library_prefix();
                    let file_extension = self.operating_system.static_library_extension();
                    let current_file_name = format!("{file_name_prefix}ryft-xla-sys-static-library.{file_extension}");
                    let new_file_name = format!("{file_name_prefix}ryft-xla-sys.{file_extension}");
                    fs::rename(
                        extracted_path.join("lib").join(current_file_name),
                        extracted_path.join("lib").join(new_file_name),
                    )
                    .with_context(|| "failed to rename the `ryft-xla-sys` static library")?;
                }
                Artifact::PjrtPlugin => {
                    let new_file = extracted_path.join(self.pjrt_plugin_library_file_name());
                    let current_file = match self.device {
                        Device::Tpu => extracted_path.join("libtpu").join("libtpu.so"),
                        Device::Neuron => extracted_path.join("libneuronxla").join("libneuronpjrt.so"),
                        Device::Metal => {
                            extracted_path.join("jax_plugins").join("metal_plugin").join("pjrt_plugin_metal_14.dylib")
                        }
                        _ => new_file.clone(),
                    };
                    if fs::exists(&current_file)? {
                        fs::rename(current_file, &new_file)
                            .with_context(|| format!("failed to rename the PJRT {} plugin library", self.device))?;
                    }
                }
            }

            Ok(extracted_path)
        }
    }

    /// Returns a string that corresponds to the target platform of this [`BuildConfiguration`]
    /// and which is primarily used for naming files (e.g., `linux-x86_64-cpu`).
    fn platform_string(&self) -> String {
        format!("{}-{}-{}", self.operating_system, self.architecture, self.device)
    }

    /// Returns the path to the requested [`Artifact`] for this [`BuildConfiguration`] set via environment variables,
    /// if present. Specifically, this function looks for environment variables like [`RYFT_XLA_SYS_ARCHIVE`],
    /// [`PJRT_PLUGIN_CUDA_12_LIB`], [`PJRT_PLUGIN_CUDA_13_LIB`], [`PJRT_PLUGIN_ROCM_7_LIB`], [`PJRT_PLUGIN_TPU_LIB`],
    /// [`PJRT_PLUGIN_NEURON_LIB`], [`PJRT_PLUGIN_METAL_LIB`] and returns the appropriate path, if present.
    fn artifact_path_from_environment(&self, artifact: Artifact) -> Result<PathBuf> {
        let artifact_name = artifact.name();

        let environment_variable = match (artifact, self.device) {
            (Artifact::RyftXlaSys, _) => Some(RYFT_XLA_SYS_ARCHIVE),
            (Artifact::PjrtPlugin, Device::Cuda12) => Some(PJRT_PLUGIN_CUDA_12_LIB),
            (Artifact::PjrtPlugin, Device::Cuda13) => Some(PJRT_PLUGIN_CUDA_13_LIB),
            (Artifact::PjrtPlugin, Device::Rocm7) => Some(PJRT_PLUGIN_ROCM_7_LIB),
            (Artifact::PjrtPlugin, Device::Tpu) => Some(PJRT_PLUGIN_TPU_LIB),
            (Artifact::PjrtPlugin, Device::Neuron) => Some(PJRT_PLUGIN_NEURON_LIB),
            (Artifact::PjrtPlugin, Device::Metal) => Some(PJRT_PLUGIN_METAL_LIB),
            _ => None,
        };

        if let Some(environment_variable) = environment_variable {
            let path = env::var(environment_variable).ok().map(|path| PathBuf::from(path));
            if let Some(path) = path {
                println!(
                    "cargo:warning=Using the `{artifact_name}` artifact specified \
                    in the `{environment_variable}` environment variable: {}.",
                    path.display(),
                );
                Ok(path)
            } else {
                Err(anyhow!(
                    "no path provided for the `{artifact_name}` artifact via \
                    the `{environment_variable}` environment variable",
                ))
            }
        } else {
            Err(anyhow!("failed to obtain the `{artifact_name}` artifact from the current environment"))
        }
    }

    /// Downloads a precompiled archive for the requested [`Artifact`] into a local cache directory if such a
    /// precompiled archive exists for this [`BuildConfiguration`].
    fn download_precompiled_artifact_archive(&self, artifact: Artifact) -> Result<PathBuf> {
        let artifact_name = artifact.name();
        let cache_dir = dirs::cache_dir()
            .map(|cache_dir| cache_dir.join("ryft").join("xla"))
            .unwrap_or(Path::new(&env::var("OUT_DIR").with_context(|| "`OUT_DIR` not set")?).join("cache"));
        let downloads_path = cache_dir.join("downloads");
        let url_prefix = self.precompiled_artifact_url_prefix(artifact);
        let archive_name = self.precompiled_artifact_name(artifact);
        let archive_download_path = downloads_path.join(archive_name.as_str());
        let url = Url::parse(format!("{url_prefix}/{archive_name}").as_str())?;
        let checksum = self.precompiled_artifact_checksum(artifact);
        let url_with_checksum = checksum.map(|checksum| UrlWithChecksum { url, checksum: checksum.to_string() });
        if let Some(url) = url_with_checksum {
            let download_result = url.download(&archive_download_path);
            if download_result.is_ok() {
                Ok(archive_download_path)
            } else {
                Err(anyhow!(
                    "failed to download a precompiled `{artifact_name}` artifact for the current \
                    build configuration due to the following reason: {download_result:?}",
                ))
            }
        } else {
            Err(anyhow!("no precompiled `{artifact_name}` artifact found for this build configuration ({self})"))
        }
    }

    /// Builds the requested [`Artifact`] using Bazel for this [`BuildConfiguration`] and returns a [`PathBuf`]
    /// pointing to the built [`Artifact`] compressed into a `.tar.gz` archive.
    fn build_artifact(&self, artifact: Artifact) -> Result<PathBuf> {
        println!("cargo::warning=Starting `{}` compilation using Bazel...", artifact.name());

        let current_path = PathBuf::from(env::current_dir().with_context(|| "Failed to get the current directory.")?);
        let output_path = PathBuf::from(env::var("OUT_DIR").with_context(|| "`OUT_DIR` not set")?);

        // Copy the Bazel workspace files to the output directory.
        // Also, monitor when they change to determine when a rebuild is necessary.
        let bazel_files = vec![
            PathBuf::from("bazel").join("archive.bzl"),
            PathBuf::from("bazel").join("BUILD.bazel"),
            PathBuf::from(".bazelrc"),
            PathBuf::from(".bazelversion"),
            PathBuf::from("BUILD.bazel"),
            PathBuf::from("pjrt_plugin.def"),
            PathBuf::from("pjrt_plugin_exported_symbols.txt"),
            PathBuf::from("pjrt_plugin_version_script.lds"),
            PathBuf::from("WORKSPACE"),
        ];

        for file_name in bazel_files {
            let source_file = current_path.join(&file_name);
            let target_file = output_path.join(&file_name);
            let target_directory = target_file.parent().unwrap();
            if let Err(error) = fs::create_dir_all(&target_directory) {
                bail!("failed to create {}; {error}", target_directory.display());
            }
            if let Err(error) = fs::copy(&source_file, &target_file) {
                bail!("failed to copy {} to {}; {error}", source_file.display(), target_file.display());
            }
        }

        let bazel_config = match &self.device {
            Device::Cpu => format!("--config={}", self.operating_system),
            Device::Cuda12 => format!("--config={} --config=cuda-12", self.operating_system),
            Device::Cuda13 => format!("--config={} --config=cuda-13", self.operating_system),
            Device::Rocm7 => format!("--config={} --config=rocm-7", self.operating_system),
            Device::Tpu | Device::Neuron | Device::Metal => {
                bail!("the PJRT {} plugin is closed source and does not support Bazel compilation", self.device)
            }
        };

        let bazel_target = match &artifact {
            Artifact::RyftXlaSys => "//:ryft-xla-sys-archive",
            Artifact::PjrtPlugin => "//:pjrt-gpu-plugin",
        };

        let status = Command::new("bazel")
            .current_dir(&output_path)
            .arg("build")
            .arg(bazel_config)
            .arg("--verbose_failures")
            .arg(bazel_target)
            .status()?;

        if !status.success() {
            bail!("failed to build {}: '{status}'", artifact.name());
        }

        println!("cargo::warning=Compiled `{}` successfully using Bazel.", artifact.name());

        let artifact_file_name = match &artifact {
            Artifact::RyftXlaSys => "ryft-xla-sys-archive.tar.gz".to_string(),
            Artifact::PjrtPlugin => {
                // For PJRT plugins, we need to create a `.tar.gz` file that contains the plugin library to match the
                // format that this function is expected to return (and in which the precompiled binaries are provided
                // in, when downloaded from the official releases page).
                let output_archive_name = format!("pjrt-plugin-{}.tar.gz", self.device);
                let output_archive_path = output_path.join(&output_archive_name);
                let output_archive = File::create(output_archive_path)?;
                let encoder = GzEncoder::new(output_archive, Compression::default());
                let mut tar = tar::Builder::new(encoder);
                let plugin_library_file_name = match self.device {
                    Device::Cpu => format!(
                        "{}pjrt-cpu-plugin.{}",
                        self.operating_system.library_prefix(),
                        self.operating_system.dynamic_library_extension(),
                    ),
                    Device::Cuda12 | Device::Cuda13 | Device::Rocm7 => format!(
                        "{}pjrt-gpu-plugin.{}",
                        self.operating_system.library_prefix(),
                        self.operating_system.dynamic_library_extension(),
                    ),
                    Device::Tpu | Device::Neuron | Device::Metal => {
                        bail!(
                            "the PJRT {} plugin is closed source and does not support Bazel compilation",
                            self.device,
                        );
                    }
                };
                let plugin_library_path = output_path.join("bazel-bin").join(plugin_library_file_name);
                let mut plugin_library_file = File::open(&plugin_library_path)?;
                tar.append_file(self.pjrt_plugin_library_file_name(), &mut plugin_library_file)?;
                tar.finish()?;
                output_archive_name
            }
        };

        Ok(output_path.join("bazel-bin").join(artifact_file_name))
    }

    /// Returns the compiled archive file name for the provided [`Artifact`] and this [`BuildConfiguration`].
    fn precompiled_artifact_name(&self, artifact: Artifact) -> String {
        match artifact {
            Artifact::RyftXlaSys => format!("ryft-xla-sys-{self}.tar.gz"),
            Artifact::PjrtPlugin => match self.device {
                Device::Tpu => "libtpu-0.0.34-cp311-cp311-manylinux_2_31_x86_64.whl".to_string(),
                Device::Neuron => "libneuronxla-2.2.14584.0%2B06ac23d1-py3-none-linux_x86_64.whl".to_string(),
                Device::Metal => "jax_metal-0.1.1-py3-none-macosx_13_0_arm64.whl".to_string(),
                _ => format!("pjrt-plugin-{}.tar.gz", self.platform_string()),
            },
        }
    }

    /// Returns the URL prefix to use for downloading the pre-compiled archive that corresponds to the provided
    /// [`Artifact`] and this [`BuildConfiguration`].
    fn precompiled_artifact_url_prefix(&self, artifact: Artifact) -> &'static str {
        match (artifact, self.device) {
            (Artifact::PjrtPlugin, Device::Tpu) => {
                "https://files.pythonhosted.org/packages/17/b9/76527052aa583529fe0b816e6bbe9010676a87e8c50da3a9751d5f404c66"
            }
            (Artifact::PjrtPlugin, Device::Neuron) => "https://pip.repos.neuron.amazonaws.com/libneuronxla",
            (Artifact::PjrtPlugin, Device::Metal) => {
                "https://files.pythonhosted.org/packages/09/dc/6d8fbfc29d902251cf333414cf7dcfaf4b252a9920c881354584ed36270d"
            }
            _ => {
                "https://github.com/eaplatanios/ryft/releases/download/ryft-xla-sys-41a5d385fedd4777e170b607e68295826fc777a8"
            }
        }
    }

    /// Returns the SHA-256 checksum for the pre-compiled archive that corresponds to the provided [`Artifact`]
    /// and this [`BuildConfiguration`].
    fn precompiled_artifact_checksum(&self, artifact: Artifact) -> Option<&'static str> {
        match (artifact, self.operating_system, self.architecture, self.device) {
            (Artifact::RyftXlaSys, OperatingSystem::Linux, Architecture::X86_64, Device::Cpu) => {
                Some("fec3b3da15179b9f6fd35a6d49261d9cbb832455e638b77bf0639f38425644d4")
            }
            (Artifact::RyftXlaSys, OperatingSystem::MacOS, Architecture::AArch64, Device::Cpu) => {
                Some("8c5d35146ae914e128c885b8eacfe03168f70818a16b2fec42e08cc111f1588d")
            }
            (Artifact::RyftXlaSys, OperatingSystem::Windows, Architecture::X86_64, Device::Cpu) => {
                Some("24617176efa50503dc8ece50bbd861665c74f9fcd26ee5ddf3ea7e3c2687959b")
            }
            (Artifact::PjrtPlugin, OperatingSystem::Linux, Architecture::X86_64, Device::Cuda12) => {
                Some("6a6150a5bd9dd0820c0cc80293b49dd030cf3debdda0873ba5946632910c58cc")
            }
            (Artifact::PjrtPlugin, OperatingSystem::Linux, Architecture::X86_64, Device::Cuda13) => {
                Some("355dfdde0bebbbeb65e9e1a0453ee54b1f4aa06f9ad7ab1100f0e95c6b76a5ce")
            }
            (Artifact::PjrtPlugin, OperatingSystem::Linux, Architecture::X86_64, Device::Rocm7) => {
                Some("32cbb8ac52917bdbd423243068a7ca64a5566f475d36b9276875c1240e10ad75")
            }
            (Artifact::PjrtPlugin, OperatingSystem::Linux, Architecture::X86_64, Device::Tpu) => {
                Some("5e600d7797ac801d0c903f52ae46c03538bb77817a48579aa581faa8d2a8a734")
            }
            (Artifact::PjrtPlugin, OperatingSystem::Linux, Architecture::X86_64, Device::Neuron) => {
                Some("d1e594b27716bc59b937ccd8f40e7f2b74f6c309643e83dcf511b7ea392924f2")
            }
            (Artifact::PjrtPlugin, OperatingSystem::MacOS, Architecture::AArch64, Device::Metal) => {
                Some("f1dbfecb298cdd3ba6da3ad6dc9a2adb63d71741f8b8ece28c296b32d608b6c8")
            }
            _ => None,
        }
    }
}

impl Display for BuildConfiguration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.platform_string())
    }
}

fn main() {
    // Skip linking to our XLA dependencies if this is executed from within a `docs.rs` pipeline.
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo::rerun-if-changed=bazel/archive.bzl");
    println!("cargo::rerun-if-changed=bazel/BUILD.bazel");
    println!("cargo::rerun-if-changed=.bazelrc");
    println!("cargo::rerun-if-changed=.bazelversion");
    println!("cargo::rerun-if-changed=BUILD.bazel");
    println!("cargo::rerun-if-changed=pjrt_plugin.def");
    println!("cargo::rerun-if-changed=pjrt_plugin_exported_symbols.txt");
    println!("cargo::rerun-if-changed=pjrt_plugin_version_script.lds");
    println!("cargo::rerun-if-changed=WORKSPACE");

    let build_configuration = BuildConfiguration::from_environment().unwrap();

    build_configuration.configure_ryft_xla_sys();

    if cfg!(feature = "cuda-12") {
        build_configuration.configure_pjrt_plugin(Device::Cuda12);
    }

    if cfg!(feature = "cuda-13") {
        build_configuration.configure_pjrt_plugin(Device::Cuda13);
    }

    if cfg!(feature = "rocm-7") {
        build_configuration.configure_pjrt_plugin(Device::Rocm7);
    }

    if cfg!(feature = "tpu") {
        build_configuration.configure_pjrt_plugin(Device::Tpu);
    }

    if cfg!(feature = "neuron") {
        build_configuration.configure_pjrt_plugin(Device::Neuron);
    }

    if cfg!(feature = "metal") {
        build_configuration.configure_pjrt_plugin(Device::Metal);
    }
}
