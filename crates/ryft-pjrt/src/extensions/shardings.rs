use prost::Message;

use crate::protos::OpSharding;
use crate::{Api, Client, Error, Executable, Plugin, invoke_pjrt_api_error_fn, slice_from_c_api};

/// The PJRT shardings extension provides capabilities for retrieving the input and output shardings of compiled
/// [`Executable`]s. The extension is both optional for PJRT [`Plugin`]s and _experimental_, meaning that incompatible
/// changes may be introduced at any time, including changes that break _Application Binary Interface (ABI)_
/// compatibility.
#[derive(Copy, Clone)]
pub struct ShardingsExtension {
    /// Handle that represents this [`ShardingsExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_Shardings_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl ShardingsExtension {
    /// Constructs a new [`ShardingsExtension`] from the provided
    /// [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base) handle if the type of that PJRT
    /// extension matches the PJRT shardings extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_Shardings {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_Shardings_Extension`](ffi::PJRT_Shardings_Extension) that corresponds
    /// to this [`ShardingsExtension`] and which can be passed to functions in the PJRT C API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_Shardings_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for ShardingsExtension {}
unsafe impl Sync for ShardingsExtension {}

impl Executable {
    /// Returns [`OpSharding`]s for all inputs of this [`Executable`]. If the loaded PJRT [`Plugin`] provides
    /// a [`ShardingsExtension`] but does not support reporting input shardings, this method will return `Ok(None)`.
    pub fn input_shardings(&self) -> Result<Option<Vec<OpSharding>>, Error> {
        use ffi::PJRT_Shardings_PJRT_Executable_ParameterShardings_Args;
        let extension = self.api().shardings_extension()?;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_Shardings_Extension => extension,
            PJRT_Shardings_PJRT_Executable_ParameterShardings,
            { executable = self.to_c_api() },
            { num_parameters, shardings, sharding_sizes },
        )
        .and_then(|(sharding_count, shardings, sharding_sizes)| {
            decode_shardings("input", shardings, sharding_sizes, sharding_count)
        })
    }

    /// Returns [`OpSharding`]s for all outputs of this [`Executable`]. If the loaded PJRT [`Plugin`] provides
    /// a [`ShardingsExtension`] but does not support reporting output shardings, this method will return `Ok(None)`.
    pub fn output_shardings(&self) -> Result<Option<Vec<OpSharding>>, Error> {
        use ffi::PJRT_Shardings_PJRT_Executable_OutputShardings_Args;
        let extension = self.api().shardings_extension()?;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_Shardings_Extension => extension,
            PJRT_Shardings_PJRT_Executable_OutputShardings,
            { executable = self.to_c_api() },
            { num_outputs, shardings, sharding_sizes },
        )
        .and_then(|(sharding_count, shardings, sharding_sizes)| {
            decode_shardings("output", shardings, sharding_sizes, sharding_count)
        })
    }
}

impl Client<'_> {
    /// Attempts to load the [`ShardingsExtension`] from this [`Client`] and returns
    /// [`Error::Unimplemented`] if it is not provided by the underlying [`Plugin`].
    pub fn shardings_extension(&self) -> Result<ShardingsExtension, Error> {
        self.api().shardings_extension()
    }
}

impl Plugin {
    /// Attempts to load the [`ShardingsExtension`] from this [`Plugin`] and returns
    /// [`Error::Unimplemented`] if it is not provided by this [`Plugin`].
    pub fn shardings_extension(&self) -> Result<ShardingsExtension, Error> {
        self.api().shardings_extension()
    }
}

impl Api {
    /// Attempts to load the [`ShardingsExtension`] from this [`Api`] and returns
    /// [`Error::Unimplemented`] if it is not provided by the underlying [`Plugin`].
    pub(crate) fn shardings_extension(&self) -> Result<ShardingsExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let shardings_extension = ShardingsExtension::from_c_api(extension, *self);
                if let Some(shardings_extension) = shardings_extension {
                    return Ok(shardings_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the shardings extension is not provided by the PJRT plugin"))
        }
    }
}

/// Internal helper that decodes an array of serialized [`OpSharding`] returned from the PJRT C API.
fn decode_shardings(
    sharding_kind: &str,
    shardings: *const *const std::ffi::c_char,
    sharding_sizes: *const usize,
    sharding_count: usize,
) -> Result<Option<Vec<OpSharding>>, Error> {
    if shardings.is_null() && sharding_sizes.is_null() {
        return Ok(None);
    }

    if shardings.is_null() || sharding_sizes.is_null() {
        return Err(Error::invalid_argument(format!(
            "encountered inconsistent PJRT executable {sharding_kind} sharding pointers: `shardings` \
            and `sharding_sizes` must either both be null or both be non-null",
        )));
    }

    let shardings = unsafe { slice_from_c_api(shardings, sharding_count) };
    let sharding_sizes = unsafe { slice_from_c_api(sharding_sizes, sharding_count) };

    let mut decoded_shardings = Vec::with_capacity(sharding_count);
    for (index, (sharding, sharding_size)) in shardings.iter().zip(sharding_sizes.iter()).enumerate() {
        if sharding.is_null() && *sharding_size > 0 {
            return Err(Error::invalid_argument(format!(
                "received a null pointer for PJRT executable {sharding_kind} sharding #{index} \
                with non-zero size {sharding_size}",
            )));
        }

        decoded_shardings.push(
            OpSharding::decode(unsafe { slice_from_c_api(*sharding as *const u8, *sharding_size) }).map_err(
                |error| {
                    Error::invalid_argument(format!(
                        "failed to deserialize PJRT executable {sharding_kind} sharding #{index} protobuf \
                returned by PJRT plugin with error: {error}",
                    ))
                },
            )?,
        );
    }

    Ok(Some(decoded_shardings))
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;
    use crate::programs::ffi::PJRT_Executable;

    pub const PJRT_API_SHARDINGS_EXTENSION_VERSION: usize = 1;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct PJRT_Shardings_PJRT_Executable_ParameterShardings_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub num_parameters: usize,
        pub shardings: *const *const std::ffi::c_char,
        pub sharding_sizes: *const usize,
    }

    impl PJRT_Shardings_PJRT_Executable_ParameterShardings_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                num_parameters: 0,
                shardings: std::ptr::null(),
                sharding_sizes: std::ptr::null(),
            }
        }
    }

    pub type PJRT_Shardings_PJRT_Executable_ParameterShardings =
        unsafe extern "C" fn(args: *mut PJRT_Shardings_PJRT_Executable_ParameterShardings_Args) -> *mut PJRT_Error;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct PJRT_Shardings_PJRT_Executable_OutputShardings_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub num_outputs: usize,
        pub shardings: *const *const std::ffi::c_char,
        pub sharding_sizes: *const usize,
    }

    impl PJRT_Shardings_PJRT_Executable_OutputShardings_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                num_outputs: 0,
                shardings: std::ptr::null(),
                sharding_sizes: std::ptr::null(),
            }
        }
    }

    pub type PJRT_Shardings_PJRT_Executable_OutputShardings =
        unsafe extern "C" fn(args: *mut PJRT_Shardings_PJRT_Executable_OutputShardings_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Shardings_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_Shardings_PJRT_Executable_ParameterShardings:
            Option<PJRT_Shardings_PJRT_Executable_ParameterShardings>,
        pub PJRT_Shardings_PJRT_Executable_OutputShardings: Option<PJRT_Shardings_PJRT_Executable_OutputShardings>,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;

    use crate::protos::{CompilationOptions, ExecutableCompilationOptions, OpShardingType, Precision};
    use crate::tests::{TestPlatform, test_cpu_client, test_for_each_platform};
    use crate::{Error, Program};

    #[test]
    fn test_shardings_extension() {
        test_for_each_platform!(|plugin, platform| {
            match platform {
                TestPlatform::Cpu | TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    assert!(plugin.shardings_extension().is_ok());
                }
                _ => {
                    assert!(matches!(plugin.shardings_extension(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }

    #[test]
    fn test_executable_shardings() {
        let options = CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: 2,
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
        };

        // Test with a program that does not have any shardings specified.
        let program = Program::Mlir {
            bytecode: indoc! {"
                module {
                    func.func @main(%arg0: tensor<2x1xi32>, %arg1: tensor<2x1xi32>) -> tensor<2x1xi32> {
                        %0 = stablehlo.add %arg0, %arg1 : tensor<2x1xi32>
                        return %0 : tensor<2x1xi32>
                    }
                }
            "}
            .as_bytes()
            .to_vec(),
        };
        let client = test_cpu_client();
        let executable = client.compile(&program, &options).unwrap();
        let executable = executable.executable().unwrap();
        let input_shardings = executable.input_shardings().unwrap().unwrap();
        let output_shardings = executable.output_shardings().unwrap().unwrap();
        assert_eq!(input_shardings.len(), 2);
        assert_eq!(input_shardings[0].r#type(), OpShardingType::Replicated);
        assert_eq!(input_shardings[1].r#type(), OpShardingType::Replicated);
        assert_eq!(output_shardings.len(), 1);
        assert_eq!(output_shardings[0].r#type(), OpShardingType::Replicated);

        // Test with a program that has both input and output shardings specified.
        let program = Program::Mlir {
            bytecode: indoc! {"
                module {
                    sdy.mesh @mesh = <[\"x\"=2]>
                    func.func @main(
                        %arg0: tensor<2x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>},
                        %arg1: tensor<2x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}
                    ) -> (tensor<2x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>}) {
                        %0 = stablehlo.add %arg0, %arg1 : tensor<2x1xi32>
                        return %0 : tensor<2x1xi32>
                    }
                }
            "}
            .as_bytes()
            .to_vec(),
        };
        let executable = client.compile(&program, &options).unwrap();
        let executable = executable.executable().unwrap();
        let input_shardings = executable.input_shardings().unwrap().unwrap();
        let output_shardings = executable.output_shardings().unwrap().unwrap();
        assert_eq!(input_shardings.len(), 2);
        assert_eq!(input_shardings[0].r#type(), OpShardingType::Other);
        assert_eq!(input_shardings[1].r#type(), OpShardingType::Replicated);
        assert_eq!(output_shardings.len(), 1);
        assert_eq!(output_shardings[0].r#type(), OpShardingType::Other);
    }
}
