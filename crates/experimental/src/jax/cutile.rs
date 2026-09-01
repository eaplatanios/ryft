use std::collections::HashMap;
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::{env, fs};

use ryft_cuda::{
    CudaArtifactFormat, CudaError, CudaKernelAbi, CudaKernelArgument, CudaKernelArtifact, CudaKernelLaunchDimensions,
    CudaKernelLauncher, CudaKernelParameterType, CudaScalarType, CudaScalarValue,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::extensions::ffi::{
    FfiBuffer, FfiBufferType, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiHandlerTraits, FfiInput,
    FfiOutput, FfiTypeId, XLA_FFI_CallFrame, XLA_FFI_Error,
};
use crate::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
use crate::tests::{TestPlatform, test_for_each_platform};
use crate::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program};

const EXPECTED_SUM: i32 = 42;
const TARGET_NAME: &str = "ryft.test.cutile.aot_add";

#[derive(Deserialize)]
struct CutileMetadata {
    schema_version: u32,
    artifact: CutileArtifactMetadata,
    kernel: CutileKernelMetadata,
    verification: CutileVerificationMetadata,
}

#[derive(Deserialize)]
struct CutileArtifactMetadata {
    format: String,
    path: String,
    sha256: String,
    size_bytes: usize,
    target_sm: String,
}

#[derive(Deserialize)]
struct CutileKernelMetadata {
    symbol: String,
    calling_convention: String,
    grid_dimensions: [u32; 3],
    block_dimensions: [u32; 3],
    shared_memory_bytes: u32,
    parameters: Vec<CutileParameterMetadata>,
}

#[derive(Deserialize)]
struct CutileParameterMetadata {
    name: String,
    constraint: String,
    dtype: Option<String>,
    ndim: Option<u32>,
    index_dtype: Option<String>,
    shape_constant: Option<Vec<i64>>,
    stride_constant: Option<Vec<i64>>,
    value: Option<i64>,
    abi: Vec<String>,
}

#[derive(Deserialize)]
struct CutileVerificationMetadata {
    jax_cutile_call_contract: bool,
}

/// A test-only cuTile artifact loaded from the bounded AOT export tool.
struct CutileTestState {
    artifact: CudaKernelArtifact,
    launcher: CudaKernelLauncher,
}

static CUTILE_TEST_STATE: LazyLock<Mutex<Option<CutileTestState>>> = LazyLock::new(|| Mutex::new(None));

fn parse_cutile_artifact(metadata_json: &str, cubin_path: &Path, cubin: Vec<u8>) -> Result<CudaKernelArtifact, String> {
    let metadata: CutileMetadata =
        serde_json::from_str(metadata_json).map_err(|error| format!("invalid cuTile metadata JSON: {error}"))?;
    if metadata.schema_version != 1 {
        return Err(format!("unsupported cuTile metadata schema version `{}`", metadata.schema_version));
    }
    if metadata.artifact.format != "cubin" {
        return Err(format!("unsupported cuTile artifact format `{}`", metadata.artifact.format));
    }
    if Path::new(metadata.artifact.path.as_str()) != cubin_path {
        return Err(format!(
            "cuTile metadata names artifact `{}`, but `RYFT_CUTILE_CUBIN` names `{}`",
            metadata.artifact.path,
            cubin_path.display(),
        ));
    }
    if metadata.artifact.size_bytes != cubin.len() {
        return Err(format!(
            "cuTile metadata records {} artifact bytes, but the cubin contains {}",
            metadata.artifact.size_bytes,
            cubin.len(),
        ));
    }
    let actual_sha256 = format!("{:x}", Sha256::digest(cubin.as_slice()));
    if metadata.artifact.sha256 != actual_sha256 {
        return Err(format!(
            "cuTile cubin SHA-256 mismatch: metadata records `{}`, but the artifact hashes to `{actual_sha256}`",
            metadata.artifact.sha256,
        ));
    }
    let Some(architecture) = metadata.artifact.target_sm.strip_prefix("sm_") else {
        return Err(format!("invalid cuTile target architecture `{}`", metadata.artifact.target_sm));
    };
    if architecture.is_empty() || !architecture.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("invalid cuTile target architecture `{}`", metadata.artifact.target_sm));
    }
    if !metadata.verification.jax_cutile_call_contract {
        return Err("cuTile metadata does not record a successful JAX calling-convention verification".to_string());
    }

    if metadata.kernel.calling_convention != "cutile_python_v2" {
        return Err(format!("unsupported cuTile calling convention `{}`", metadata.kernel.calling_convention,));
    }
    let expected_arrays = ["lhs", "rhs", "output"];
    let mut array_index = 0;
    let mut parameters = Vec::new();
    for parameter in &metadata.kernel.parameters {
        match parameter.constraint.as_str() {
            "ArrayConstraint" => {
                let expected_name = expected_arrays.get(array_index).ok_or_else(|| {
                    format!("cuTile metadata contains unexpected array parameter `{}`", parameter.name)
                })?;
                if parameter.name != *expected_name
                    || parameter.dtype.as_deref() != Some("int32")
                    || parameter.ndim != Some(1)
                    || parameter.index_dtype.as_deref() != Some("int32")
                    || parameter.shape_constant.as_deref() != Some(&[1])
                    || parameter.stride_constant.as_deref() != Some(&[1])
                    || parameter.abi != ["device_pointer", "shape_i32", "stride_i32"]
                {
                    return Err(format!(
                        "cuTile array parameter `{}` does not match the vector-add seam contract",
                        parameter.name,
                    ));
                }
                for abi_parameter in &parameter.abi {
                    parameters.push(match abi_parameter.as_str() {
                        "device_pointer" => CudaKernelParameterType::DevicePointer,
                        "shape_i32" | "stride_i32" => CudaKernelParameterType::Scalar(CudaScalarType::I32),
                        _ => return Err(format!("unsupported cuTile ABI parameter `{abi_parameter}`")),
                    });
                }
                array_index += 1;
            }
            "ConstantConstraint" => {
                if parameter.name != "tile_size" || parameter.value != Some(1) || !parameter.abi.is_empty() {
                    return Err(format!(
                        "cuTile constant parameter `{}` does not match the vector-add seam contract",
                        parameter.name,
                    ));
                }
            }
            constraint => return Err(format!("unsupported cuTile parameter constraint `{constraint}`")),
        }
    }
    if array_index != expected_arrays.len() {
        return Err(format!(
            "cuTile metadata contains {array_index} array parameters, expected {}",
            expected_arrays.len(),
        ));
    }

    let dimensions = CudaKernelLaunchDimensions::new(
        metadata.kernel.grid_dimensions,
        metadata.kernel.block_dimensions,
        metadata.kernel.shared_memory_bytes,
    )
    .map_err(|error| format!("invalid cuTile launch dimensions: {error}"))?;
    let abi = CudaKernelAbi::new(metadata.kernel.calling_convention, 2, parameters)
        .map_err(|error| format!("invalid cuTile kernel ABI: {error}"))?;
    CudaKernelArtifact::new(
        CudaArtifactFormat::Cubin,
        cubin,
        metadata.kernel.symbol,
        metadata.artifact.target_sm,
        dimensions,
        abi,
    )
    .map_err(|error| format!("invalid cuTile kernel artifact: {error}"))
}

fn test_compilation_options() -> CompilationOptions {
    CompilationOptions {
        argument_layouts: Vec::new(),
        parameter_is_tupled_arguments: false,
        executable_build_options: Some(ExecutableCompilationOptions {
            device_ordinal: -1,
            replica_count: 1,
            partition_count: 1,
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

fn test_program() -> Program {
    Program::Mlir {
        bytecode: format!(
            r#"
            module {{
              func.func @main(%lhs: tensor<1xi32>, %rhs: tensor<1xi32>) -> tensor<1xi32> {{
                %output = stablehlo.custom_call @"{TARGET_NAME}"(%lhs, %rhs) {{api_version = 4 : i32}}
                  : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
                return %output : tensor<1xi32>
              }}
            }}
            "#,
        )
        .into_bytes(),
    }
}

fn validate_buffer(buffer: FfiBuffer<'_>) -> Result<FfiBuffer<'_>, FfiError> {
    if buffer.element_type() != FfiBufferType::I32 || buffer.dimensions() != [1] {
        return Err(FfiError::invalid_argument("expected a rank-one, one-element `i32` buffer"));
    }
    Ok(buffer)
}

fn input_buffer(input: FfiInput<'_>) -> Result<FfiBuffer<'_>, FfiError> {
    let FfiInput::Buffer { buffer } = input;
    validate_buffer(buffer)
}

fn output_buffer(output: FfiOutput<'_>) -> Result<FfiBuffer<'_>, FfiError> {
    let FfiOutput::Buffer { buffer } = output;
    validate_buffer(buffer)
}

fn launch_cutile(call_frame: &FfiCallFrame<'_>) -> Result<(), FfiError> {
    if call_frame.input_count() != 2 || call_frame.output_count() != 1 {
        return Err(FfiError::invalid_argument("expected two inputs and one output for the cuTile seam probe"));
    }

    let context = call_frame.context()?;
    let lhs_buffer = input_buffer(call_frame.input(0)?)?;
    let rhs_buffer = input_buffer(call_frame.input(1)?)?;
    let out_buffer = output_buffer(call_frame.output(0)?)?;

    // `cutile_python_v2` flattens each rank-one ArrayConstraint into its device pointer, shape, and stride.
    // The ConstantConstraint for the tile size is intentionally omitted from the launch ABI.
    let launch = (|| -> Result<_, CudaError> {
        unsafe {
            context.cuda_kernel_launch(vec![
                lhs_buffer.cuda_kernel_argument()?,
                CudaKernelArgument::scalar(CudaScalarValue::I32(1)),
                CudaKernelArgument::scalar(CudaScalarValue::I32(1)),
                rhs_buffer.cuda_kernel_argument()?,
                CudaKernelArgument::scalar(CudaScalarValue::I32(1)),
                CudaKernelArgument::scalar(CudaScalarValue::I32(1)),
                out_buffer.cuda_kernel_argument()?,
                CudaKernelArgument::scalar(CudaScalarValue::I32(1)),
                CudaKernelArgument::scalar(CudaScalarValue::I32(1)),
            ])
        }
    })()
    .map_err(|error| FfiError::internal(format!("failed to construct cuTile launch frame: {error}")))?;

    let state = CUTILE_TEST_STATE.lock().unwrap();
    let state = state.as_ref().ok_or_else(|| FfiError::internal("cuTile seam probe has no loaded cubin fixture"))?;
    // SAFETY: The seam-test state is cleared before the PJRT client that owns this stream's CUDA context is dropped.
    unsafe { state.launcher.launch(&state.artifact, &launch) }
        .map_err(|error| FfiError::internal(format!("failed to launch cuTile seam probe: {error}")))
}

unsafe extern "C" fn cutile_handler(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
    let call_frame = match unsafe { FfiCallFrame::from_c_api(call_frame) } {
        Ok(call_frame) => call_frame,
        Err(_) => return std::ptr::null_mut(),
    };
    if call_frame.register_metadata(FfiTypeId::UNKNOWN) || call_frame.stage() != FfiExecutionStage::Execution {
        return std::ptr::null_mut();
    }

    let Err(error) = launch_cutile(&call_frame) else {
        return std::ptr::null_mut();
    };
    match call_frame.api() {
        Ok(api) => unsafe { error.to_c_api(api) },
        Err(_) => std::ptr::null_mut(),
    }
}

#[test]
fn test_cutile_cubin_on_xla_ffi_cuda_stream() {
    if env::var("RYFT_PJRT_RUN_CUTILE_SEAM_PROBE").ok().as_deref() != Some("1") {
        return;
    }

    let cubin_path = env::var("RYFT_CUTILE_CUBIN")
        .expect("`RYFT_CUTILE_CUBIN` must name the exported cuTile cubin when the seam probe is enabled");
    let cubin = fs::read(cubin_path.as_str())
        .unwrap_or_else(|error| panic!("failed to read cuTile cubin `{cubin_path}`: {error}"));
    let metadata_path = env::var("RYFT_CUTILE_METADATA")
        .expect("`RYFT_CUTILE_METADATA` must name the exported cuTile metadata when the seam probe is enabled");
    let metadata_json = fs::read_to_string(metadata_path.as_str())
        .unwrap_or_else(|error| panic!("failed to read cuTile metadata `{metadata_path}`: {error}"));
    let artifact = parse_cutile_artifact(metadata_json.as_str(), Path::new(cubin_path.as_str()), cubin)
        .unwrap_or_else(|error| panic!("failed to validate cuTile export metadata: {error}"));
    let mut tested_cuda = false;
    test_for_each_platform!(|_plugin, client, platform| {
        if platform == TestPlatform::Cuda13 {
            tested_cuda = true;
            let launcher =
                client.cuda_kernel_launcher().expect("failed to load the CUDA Driver API for the cuTile seam probe");
            *CUTILE_TEST_STATE.lock().unwrap() = Some(CutileTestState { artifact: artifact.clone(), launcher });
            let platform_name = client.platform_name().unwrap();
            assert_eq!(
                client.register_ffi_handler(
                    TARGET_NAME,
                    platform_name.as_ref(),
                    FfiHandler::from(cutile_handler as crate::extensions::ffi::XLA_FFI_Handler),
                    FfiHandlerTraits::NONE,
                ),
                Ok(()),
            );

            let executable = client.compile(&test_program(), &test_compilation_options()).unwrap();
            let device = executable.addressable_devices().unwrap().remove(0);
            let lhs = client
                .buffer(7i32.to_ne_bytes().as_slice(), BufferType::I32, &[1], None, device.clone(), None)
                .unwrap();
            let rhs = client.buffer(35i32.to_ne_bytes().as_slice(), BufferType::I32, &[1], None, device, None).unwrap();
            let execution_inputs = [ExecutionInput::from(lhs), ExecutionInput::from(rhs)];
            let inputs = ExecutionDeviceInputs::from(execution_inputs.as_slice());
            let mut device_outputs = executable
                .execute(vec![inputs], vec![], 0, None, None, None, None)
                .unwrap()
                .block_until_ready()
                .unwrap()
                .remove(0);
            let output = device_outputs.outputs.remove(0);
            let output_bytes = output.copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(output_bytes, EXPECTED_SUM.to_ne_bytes());

            // Unload cached modules before dropping the PJRT client that owns their CUDA contexts.
            let mut state = CUTILE_TEST_STATE.lock().unwrap().take().unwrap();
            // SAFETY: `client` still owns every CUDA context observed by this launcher.
            unsafe { state.launcher.shutdown() }.unwrap();
        }
    });

    assert!(tested_cuda, "cuTile seam probe requires `ryft-experimental` to be built with the `cuda-13` feature");
}

#[test]
fn test_cutile_metadata_contract() {
    let cubin = vec![1, 2, 3, 4];
    let cubin_path = Path::new("/tmp/ryft-cutile-test.cubin");
    let metadata = serde_json::json!({
        "schema_version": 1,
        "artifact": {
            "format": "cubin",
            "path": cubin_path,
            "sha256": format!("{:x}", Sha256::digest(cubin.as_slice())),
            "size_bytes": cubin.len(),
            "target_sm": "sm_100",
        },
        "kernel": {
            "symbol": "ryft_cutile_vector_add",
            "calling_convention": "cutile_python_v2",
            "grid_dimensions": [1, 1, 1],
            "block_dimensions": [1, 1, 1],
            "shared_memory_bytes": 0,
            "parameters": [
                {
                    "name": "lhs",
                    "constraint": "ArrayConstraint",
                    "dtype": "int32",
                    "ndim": 1,
                    "index_dtype": "int32",
                    "shape_constant": [1],
                    "stride_constant": [1],
                    "abi": ["device_pointer", "shape_i32", "stride_i32"],
                },
                {
                    "name": "rhs",
                    "constraint": "ArrayConstraint",
                    "dtype": "int32",
                    "ndim": 1,
                    "index_dtype": "int32",
                    "shape_constant": [1],
                    "stride_constant": [1],
                    "abi": ["device_pointer", "shape_i32", "stride_i32"],
                },
                {
                    "name": "output",
                    "constraint": "ArrayConstraint",
                    "dtype": "int32",
                    "ndim": 1,
                    "index_dtype": "int32",
                    "shape_constant": [1],
                    "stride_constant": [1],
                    "abi": ["device_pointer", "shape_i32", "stride_i32"],
                },
                {
                    "name": "tile_size",
                    "constraint": "ConstantConstraint",
                    "value": 1,
                    "abi": [],
                },
            ],
        },
        "verification": {
            "jax_cutile_call_contract": true,
        },
    });

    let artifact = parse_cutile_artifact(metadata.to_string().as_str(), cubin_path, cubin.clone()).unwrap();
    assert_eq!(artifact.bytes(), cubin);
    assert_eq!(artifact.symbol(), "ryft_cutile_vector_add");
    assert_eq!(artifact.target_architecture(), "sm_100");
    assert_eq!(artifact.launch_dimensions().grid(), [1, 1, 1]);
    assert_eq!(artifact.abi().schema(), "cutile_python_v2");
    assert_eq!(artifact.abi().version(), 2);
    assert_eq!(artifact.abi().parameters().len(), 9);

    let mut invalid_metadata = metadata.clone();
    invalid_metadata["kernel"]["calling_convention"] = serde_json::Value::String("cutile_python_v3".to_string());
    assert_eq!(
        parse_cutile_artifact(invalid_metadata.to_string().as_str(), cubin_path, cubin.clone()).unwrap_err(),
        "unsupported cuTile calling convention `cutile_python_v3`",
    );

    let mut invalid_metadata = metadata.clone();
    invalid_metadata["artifact"]["sha256"] = serde_json::Value::String("invalid".to_string());
    assert!(
        parse_cutile_artifact(invalid_metadata.to_string().as_str(), cubin_path, cubin.clone())
            .unwrap_err()
            .contains("SHA-256 mismatch"),
    );

    let mut invalid_metadata = metadata;
    invalid_metadata["kernel"]["parameters"][0]["abi"] =
        serde_json::json!(["device_pointer", "stride_i32", "shape_i32"]);
    assert!(
        parse_cutile_artifact(invalid_metadata.to_string().as_str(), cubin_path, vec![1, 2, 3, 4])
            .unwrap_err()
            .contains("does not match the vector-add seam contract"),
    );
}
