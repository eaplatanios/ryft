//! Direct Triton artifacts embedded through the existing concrete platform runtime.

use ryft_core::{CustomCallOperation, EffectClass, Typed};
use ryft_kernels::VerifiedKernel;
use ryft_triton::kernels::{Artifact, COMPILER_SCHEMA_VERSION, CompiledKernel, Target};
use sha2::{Digest, Sha256};

use crate::kernels::cuda::{CUDA_KERNEL_CUSTOM_CALL_TARGET, CudaKernelEmbedding, CudaKernelParameterBinding};
use crate::kernels::rocm::RocmKernelEmbedding;
use crate::kernels::staging::{KernelExecutionFacts, KernelTarget};
use crate::kernels::{KernelEmbeddingError, KernelOutputEmbedding, ROCM_KERNEL_CUSTOM_CALL_TARGET};

// TODO(eaplatanios): Review this module.

/// Validates direct Triton output against the logical boundary and delegates concrete artifact execution.
#[derive(Copy, Clone, Debug, Default)]
pub struct TritonEmbedding;

impl KernelOutputEmbedding<CompiledKernel> for TritonEmbedding {
    fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
        Ok(format!(
            "Triton embedding 1; compiler schema {COMPILER_SCHEMA_VERSION}; cuda envelope 1; \
             rocm envelope 1; static global pointers"
        )
        .into_bytes())
    }

    fn custom_call(
        &self,
        kernel: &VerifiedKernel<'_>,
        output: &CompiledKernel,
    ) -> Result<CustomCallOperation, KernelEmbeddingError> {
        let logical = kernel.definition().operation();
        if output.semantic_key() != kernel.definition().semantic_key()? || !logical.prefetch_types().is_empty() {
            return Err(KernelEmbeddingError::Invalid {
                message: "triton artifact differs from the specialized kernel definition".into(),
            });
        }
        let types = logical.parameters().iter().map(|parameter| parameter.r#type().into_owned()).collect::<Vec<_>>();
        if output.parameter_types() != types {
            return Err(KernelEmbeddingError::Invalid {
                message: "triton parameter types differ from the logical kernel boundary".into(),
            });
        }
        let operation = match output.artifact() {
            Artifact::Cuda(artifact) => CudaKernelEmbedding::from_parameters(
                CUDA_KERNEL_CUSTOM_CALL_TARGET.into(),
                (0..types.len()).map(CudaKernelParameterBinding::Array).collect(),
            )
            .with_row_major_layouts(true)
            .with_assertions(kernel.definition().body().effects().classes().contains(EffectClass::OrderedAssertion))
            .custom_call(kernel, artifact)?,
            Artifact::Rocm(artifact) => {
                RocmKernelEmbedding::new(ROCM_KERNEL_CUSTOM_CALL_TARGET.into()).custom_call(kernel, artifact)?
            }
        };
        Ok(operation
            .with_attribute("ryft.triton.schema", i64::from(COMPILER_SCHEMA_VERSION))
            .with_attribute("ryft.triton.configuration", hex::encode(Sha256::digest(output.configuration_key()))))
    }
}

impl KernelTarget for Target {
    fn admit_execution(&self, facts: &KernelExecutionFacts) -> Result<(), KernelEmbeddingError> {
        let capability = match self {
            Target::Cuda { major, minor } => {
                if !matches!((*major, *minor), (8, 0) | (12, 1))
                    || !facts.platform_name().eq_ignore_ascii_case("cuda")
                    || facts.platform_version() != "cuda 13020"
                    || facts.pjrt_version() != ryft_pjrt::VERSION
                    || !facts.has_ffi_extension()
                    || facts.devices().is_empty()
                {
                    return Err(KernelEmbeddingError::Invalid {
                        message: "triton requires a qualified CUDA target and the pinned CUDA 13.2 PJRT FFI platform"
                            .into(),
                    });
                }
                format!("{major}.{minor}")
            }
            Target::Rocm { architecture } => {
                // The pinned StreamExecutor client exposes the runtime major, minor, and patch versions as this decimal
                // encoding. `MakeComputeCapabilityAttributeString` returns `gfx_version()`, excluding `xnack` and
                // `sramecc` suffixes.
                // The concrete HIP launcher checks those ELF feature requirements against actual device properties.
                let version =
                    facts.platform_version().strip_prefix("rocm ").and_then(|value| value.parse::<u32>().ok());
                if !matches!(architecture.as_str(), "gfx908" | "gfx90a" | "gfx942")
                    || !facts.platform_name().eq_ignore_ascii_case("rocm")
                    || version.is_none_or(|version| !(71_300_000..71_400_000).contains(&version))
                    || facts.pjrt_version() != ryft_pjrt::VERSION
                    || !facts.has_ffi_extension()
                    || facts.devices().is_empty()
                {
                    return Err(KernelEmbeddingError::Invalid {
                        message: "triton requires a supported AMD target and the HIP 7.13 PJRT FFI platform".into(),
                    });
                }
                architecture.clone()
            }
        };
        if facts.devices().iter().any(|device| {
            device.attributes().get("compute_capability") != Some(&ryft_pjrt::Value::String(capability.clone()))
        }) {
            return Err(KernelEmbeddingError::Invalid {
                message: format!("triton requires exact device compute capability `{capability}`"),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayType, CustomCallAttribute, DataType, Device, DeviceMesh, DiskCache, LogicalMesh, StatefulExecution,
    };
    use ryft_kernels::{KernelCompilationError, KernelCompiler, KernelSchedule, kernel};
    use ryft_triton::kernels::{Compiler, Error, Options, TRITON_VERSION, XLA_VERSION, lowering_sha256};
    use serde_json::json;

    use crate::FromPjrt;
    use crate::kernels::aot::tests::executable_bundle;
    use crate::kernels::aot::{KernelAotBundle, KernelAotError};
    use crate::kernels::staging::{KernelCompilerBinding, KernelDeviceFacts};
    use crate::kernels::tuning::{
        KernelTuner, KernelTuningBudget, KernelTuningError, KernelTuningRequest, KernelTuningRunner,
    };
    use crate::sessions::tests::session_with_client;
    use crate::tests::execution_client;

    use super::*;

    #[cfg(feature = "cuda-13")]
    mod barrier_execution {
        use std::num::NonZeroUsize;

        use pretty_assertions::assert_eq;

        use ryft_core::{ArrayType, CompilationTracer, DataType, StagedFunction};
        use ryft_kernels::{KernelSchedule, kernel};
        use ryft_triton::kernels::{Compiler, Options as TritonOptions, Target};

        use crate::kernels::{KernelCompilerBinding, TritonEmbedding, stage_kernel};
        use crate::sessions::tests::domain_mesh;
        use crate::tests::execution_client;
        use crate::{Array, Options, Session, Type, Value};

        /// Writes a workgroup tile and copies a slice whose elements are assigned to different lanes.
        #[kernel(crate = "ryft_kernels", requires = input.shape()[0] == 256)]
        fn read_written_slice(
            #[input(data_type = F32, rank = 1)] input: &ryft_core::Array,
            #[output(data_type = F32, shape = [256])] intermediate: &mut ryft_core::Array,
            #[output(data_type = F32, shape = [128])] output: &mut ryft_core::Array,
        ) {
            intermediate.store(input.load());
            output.store(intermediate.slice([64..192]).load());
        }

        #[test]
        fn test_triton_embedding_executes_read_written_slice() {
            if std::env::var("RYFT_RUN_TRITON_COMPILER_TESTS").ok().as_deref() != Some("1")
                || std::env::var("RYFT_TEST_EXECUTION_PLATFORM").ok().as_deref() != Some("cuda-13")
            {
                return;
            }
            let client = execution_client();
            let device = client.addressable_devices().unwrap().remove(0);
            let ryft_pjrt::Value::String(capability) = device.attribute("compute_capability").unwrap() else {
                panic!("missing CUDA compute capability")
            };
            let (major, minor) = capability.split_once('.').unwrap();
            let target = Target::Cuda { major: major.parse().unwrap(), minor: minor.parse().unwrap() };
            let mesh = domain_mesh(client, "device", 1);
            let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
            let r#type = ArrayType::new_static(DataType::F32, [256]);
            let definition = read_written_slice::definition(&r#type).unwrap();
            let values = (0..256).map(|value| value as f32).collect::<Vec<_>>();
            let input = Array::from_host_buffer(
                &session,
                r#type.clone(),
                None,
                None,
                values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>(),
            )
            .unwrap();

            for stages in [1, 2, 3] {
                let schedule = KernelSchedule::default().with_pipeline_stages(NonZeroUsize::new(stages).unwrap());
                let binding = KernelCompilerBinding::new(
                    Compiler::new(),
                    target.clone(),
                    TritonOptions::default(),
                    schedule,
                    TritonEmbedding,
                    1024,
                )
                .unwrap();
                let staged = StagedFunction::<Session, Vec<Type>, Vec<Type>>::stage(
                    |_, _, inputs: Vec<CompilationTracer<Session>>| {
                        Ok(stage_kernel(inputs[0].context(), &definition, &inputs)?)
                    },
                    vec![],
                    vec![Type::Array(r#type.clone())],
                    Options::new(mesh.clone()).with_kernel_compiler(binding),
                )
                .unwrap();
                let compiled = staged.lower(&session).unwrap().compile(&session).unwrap();
                let outputs = compiled.executable().execute(&session, vec![Value::Array(input.clone())]).unwrap();
                let [Value::Array(intermediate), Value::Array(output)] = outputs.as_slice() else {
                    panic!("expected two array outputs")
                };
                let read_values = |array: &crate::Array| {
                    let bytes = array
                        .device_shard(device.id().unwrap())
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .copy_to_host(None)
                        .unwrap()
                        .r#await()
                        .unwrap();
                    bytes
                        .chunks_exact(size_of::<f32>())
                        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(read_values(intermediate), values);
                assert_eq!(read_values(output), values[64..192]);
            }
        }
    }

    /// Recorded schema 1 configuration of the subprocess compiler at `7197043f1`, including its protocol and
    /// executable hash. The linked compiler reuses schema 1 for a different record until the first release.
    fn historical_configuration() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema": 1,
            "versions": {
                "schema": 1, "protocol": 1, "xla": "eb6b90ed013f511eca088c52f541f3c0819f919e",
                "jax": "a7606f995e1a92707cbeb257e487fa53e7abe84b", "triton": "a77e7c793abc0d0c923a9afb275058e2fe57a198",
                "rocm_device_libs": "53996464fa8d94b182ac4aaa7dc3a109ab524f45",
                "cuda_toolkit_version": 13020, "assembler_version": "13.0.88",
            },
            "executable_sha256": "a".repeat(64), "target": Target::Cuda { major: 12, minor: 1 }, "warp_count": 4,
            "pipeline_stages": 2, "maximum_scratch_bytes": null,
        }))
        .unwrap()
    }

    /// Recorded configuration of the linked compiler for the same target and schedule as the historical record.
    fn configuration() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema": COMPILER_SCHEMA_VERSION,
            "versions": {
                "schema": COMPILER_SCHEMA_VERSION, "xla": XLA_VERSION, "jax": ryft_xla_sys::JAX_COMMIT,
                "triton": TRITON_VERSION, "rocm_device_libs": "53996464fa8d94b182ac4aaa7dc3a109ab524f45",
                "cuda_available": true, "rocm_available": true, "cuda_toolkit_version": 13020,
                "assembler_version": "13.0.88",
            },
            "lowering_sha256": lowering_sha256(), "target": Target::Cuda { major: 12, minor: 1 }, "warp_count": 4,
            "pipeline_stages": 2, "maximum_scratch_bytes": null,
        }))
        .unwrap()
    }

    /// Stands in for the producer of a recorded configuration that cannot be restored, by reporting it as its key.
    struct RecordedProducer(Vec<u8>);

    impl KernelCompiler for RecordedProducer {
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
            let _ = (kernel, target, options, schedule);
            panic!("configuration identity checks must not invoke compiler admission")
        }

        fn configuration_key(
            &self,
            target: &Target,
            options: &Options,
            schedule: &KernelSchedule,
        ) -> Result<Vec<u8>, KernelCompilationError<Error>> {
            let _ = (target, options, schedule);
            Ok(self.0.clone())
        }

        fn compile(
            &self,
            kernel: &VerifiedKernel<'_>,
            target: &Target,
            options: &Options,
            schedule: &KernelSchedule,
        ) -> Result<CompiledKernel, KernelCompilationError<Error>> {
            let _ = (kernel, target, options, schedule);
            panic!("configuration identity checks must not invoke the compiler")
        }
    }

    /// Exercises the native compiler and embedding contracts without requiring a device runtime.
    #[kernel(crate = "ryft_kernels")]
    fn vector(
        #[input(data_type = F32, rank = 1)] left: &ryft_core::Array,
        #[input(data_type = F32, rank = 1)] right: &ryft_core::Array,
        #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut ryft_core::Array,
    ) {
        output.store(left.load() + right.load());
    }

    #[test]
    fn test_triton_embedding_configuration_key() {
        assert_eq!(
            String::from_utf8(TritonEmbedding.configuration_key().unwrap()).unwrap(),
            "Triton embedding 1; compiler schema 1; cuda envelope 1; \
             rocm envelope 1; static global pointers"
        );
    }

    #[test]
    fn test_triton_embedding_custom_call() {
        if std::env::var("RYFT_RUN_TRITON_COMPILER_TESTS").ok().as_deref() != Some("1") {
            return;
        }
        let compiler = Compiler::new();
        let r#type = ArrayType::new_static(DataType::F32, [256]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let verified = VerifiedKernel::new(&definition, 1024).unwrap();
        let changed_type = ArrayType::new_static(DataType::F32, [257]);
        let changed = vector::definition(&changed_type, &changed_type).unwrap();
        let changed = VerifiedKernel::new(&changed, 1024).unwrap();
        for (target, expected_target) in [
            (Target::Cuda { major: 12, minor: 1 }, CUDA_KERNEL_CUSTOM_CALL_TARGET),
            (Target::Rocm { architecture: "gfx942".into() }, ROCM_KERNEL_CUSTOM_CALL_TARGET),
        ] {
            let compiled =
                compiler.compile(&verified, &target, &Options::default(), &KernelSchedule::default()).unwrap();
            let operation = TritonEmbedding.custom_call(&verified, &compiled).unwrap();
            assert_eq!(operation.target_name(), expected_target);
            assert_eq!(
                &operation.attributes()[operation.attributes().len() - 2..],
                &[
                    ("ryft.triton.schema".to_owned(), CustomCallAttribute::I64(i64::from(COMPILER_SCHEMA_VERSION))),
                    (
                        "ryft.triton.configuration".to_owned(),
                        CustomCallAttribute::String(hex::encode(Sha256::digest(compiled.configuration_key()))),
                    ),
                ],
            );
            match target {
                Target::Cuda { .. } => {
                    CudaKernelEmbedding::from_custom_call(&verified, &operation).unwrap();
                }
                Target::Rocm { .. } => {
                    RocmKernelEmbedding::from_custom_call(&verified, &operation).unwrap();
                }
            }
            assert!(matches!(TritonEmbedding.custom_call(&changed, &compiled),
                Err(KernelEmbeddingError::Invalid { message })
                    if message == "triton artifact differs from the specialized kernel definition"));
        }
    }

    #[test]
    fn test_kernel_aot_bundle_load_triton_historical_schema() {
        let client = execution_client();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![]).unwrap(),
            vec![Device::from_pjrt(client.addressable_devices().unwrap().remove(0)).unwrap()],
        )
        .unwrap();
        let session = session_with_client(client).with_device_mesh(mesh.clone());
        let target = Target::Cuda { major: 12, minor: 1 };
        assert!(matches!(Compiler::from_configuration(&historical_configuration()),
            Err(Error::Invalid { message }) if message == "invalid recorded compiler configuration"));
        let historical = KernelCompilerBinding::new(
            RecordedProducer(historical_configuration()),
            target.clone(),
            Options::default(),
            KernelSchedule::default(),
            TritonEmbedding,
            1,
        )
        .unwrap();
        let fresh = KernelCompilerBinding::new(
            Compiler::from_configuration(&configuration()).unwrap(),
            target,
            Options::default(),
            KernelSchedule::default(),
            TritonEmbedding,
            1,
        )
        .unwrap();

        // A bundle from the historical producer differs only in its compiler configuration, and the current binding
        // rejects it before the executable is deserialized.
        let bundle = executable_bundle(&session, &mesh, DataType::I32, false, &historical);
        let bundle = KernelAotBundle::from_bytes(&bundle.to_bytes().unwrap(), 1).unwrap();
        bundle.load(&session, &historical, &mesh).unwrap();
        assert!(matches!(
            bundle.load(&session, &fresh, &mesh),
            Err(KernelAotError::Invalid { message }) if message == "compiler binding configuration mismatch",
        ));

        // A bundle from the current producer loads with the current binding.
        let bundle = executable_bundle(&session, &mesh, DataType::I32, false, &fresh);
        let bundle = KernelAotBundle::from_bytes(&bundle.to_bytes().unwrap(), 1).unwrap();
        bundle.load(&session, &fresh, &mesh).unwrap();
    }

    #[test]
    fn test_kernel_tuner_load_triton_historical_schema() {
        /// Completes each invocation synchronously, because only the identity of persisted records is under test.
        struct Runner;

        impl KernelTuningRunner for Runner {
            fn prepare(&mut self, schedule: &KernelSchedule) -> Result<(), KernelTuningError> {
                let _ = schedule;
                Ok(())
            }

            fn execute(&mut self) -> StatefulExecution<(), KernelTuningError> {
                StatefulExecution::ready(Ok(()))
            }
        }

        let directory = tempfile::tempdir().unwrap();
        let tuner = KernelTuner::new(Some(Arc::new(DiskCache::open(directory.path()).unwrap())));
        let r#type = ArrayType::new_static(DataType::F32, [256]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let kernel = VerifiedKernel::new(&definition, 1024).unwrap();
        let target = Target::Cuda { major: 12, minor: 1 };
        let facts = KernelExecutionFacts::new(
            "CUDA",
            "cuda 13020",
            ryft_pjrt::VERSION,
            true,
            BTreeMap::new(),
            vec![KernelDeviceFacts::new(
                "GB10",
                BTreeMap::from([("compute_capability".into(), ryft_pjrt::Value::String("12.1".into()))]),
            )],
        );
        let budget = KernelTuningBudget::new(1, 0, 1, Duration::from_secs(10)).unwrap();
        let historical = KernelTuningRequest::new(
            &kernel,
            &RecordedProducer(historical_configuration()),
            &target,
            &Options::default(),
            &TritonEmbedding,
            &facts,
            b"vector-256",
            vec![KernelSchedule::default()],
            budget.clone(),
        )
        .unwrap();
        let fresh = KernelTuningRequest::new(
            &kernel,
            &Compiler::from_configuration(&configuration()).unwrap(),
            &target,
            &Options::default(),
            &TritonEmbedding,
            &facts,
            b"vector-256",
            vec![KernelSchedule::default()],
            budget,
        )
        .unwrap();

        // A measurement persisted for the historical producer misses the current request.
        let stale = tuner.run(&historical, &mut Runner, &AtomicBool::new(false)).unwrap();
        assert_eq!(tuner.load(&historical).unwrap(), Some(stale));
        assert_eq!(tuner.load(&fresh).unwrap(), None);

        // A cold measurement for the current producer is reused by a warm load.
        let result = tuner.run(&fresh, &mut Runner, &AtomicBool::new(false)).unwrap();
        assert_eq!(tuner.load(&fresh).unwrap(), Some(result));
    }

    #[test]
    fn test_target_admit_execution() {
        let facts = |has_ffi_extension: bool| {
            KernelExecutionFacts::new(
                "CUDA",
                "cuda 13020",
                ryft_pjrt::VERSION,
                has_ffi_extension,
                BTreeMap::new(),
                vec![KernelDeviceFacts::new(
                    "GB10",
                    BTreeMap::from([("compute_capability".into(), ryft_pjrt::Value::String("12.1".into()))]),
                )],
            )
        };
        let target = Target::Cuda { major: 12, minor: 1 };
        target.admit_execution(&facts(true)).unwrap();
        let result = Target::Cuda { major: 8, minor: 0 }.admit_execution(&facts(true));
        assert!(matches!(result, Err(KernelEmbeddingError::Invalid { message })
            if message == "triton requires exact device compute capability `8.0`"));
        assert!(matches!(target.admit_execution(&facts(false)), Err(KernelEmbeddingError::Invalid { message })
            if message == "triton requires a qualified CUDA target and the pinned CUDA 13.2 PJRT FFI platform"));
    }

    #[test]
    fn test_target_admit_execution_rocm() {
        let facts = |platform_version: &str, architecture: &str| {
            KernelExecutionFacts::new(
                "ROCM",
                platform_version,
                ryft_pjrt::VERSION,
                true,
                BTreeMap::new(),
                vec![KernelDeviceFacts::new(
                    "AMD Instinct MI300X",
                    BTreeMap::from([("compute_capability".into(), ryft_pjrt::Value::String(architecture.into()))]),
                )],
            )
        };
        let target = Target::Rocm { architecture: "gfx942".into() };
        target.admit_execution(&facts("rocm 71300000", "gfx942")).unwrap();
        let result = target.admit_execution(&facts("rocm 71300000", "gfx90a"));
        assert!(matches!(result, Err(KernelEmbeddingError::Invalid { message })
            if message == "triton requires exact device compute capability `gfx942`"));
        let result = target.admit_execution(&facts("rocm 71400000", "gfx90a"));
        assert!(matches!(result, Err(KernelEmbeddingError::Invalid { message })
            if message == "triton requires a supported AMD target and the HIP 7.13 PJRT FFI platform"));
    }
}
