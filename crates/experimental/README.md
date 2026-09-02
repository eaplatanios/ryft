# Ryft Experimental

This unpublished crate contains test-only integration seam probes. It does not expose supported runtime APIs.

The cuTile CUDA probe exports a cubin and a JSON metadata record with `tools/cutile/export_vector_add.py`. The test
consumes both `RYFT_CUTILE_CUBIN` and `RYFT_CUTILE_METADATA`, validates the artifact checksum, ELF architecture, and
launch contract, launches it through `ryft-cuda` using the production `ryft-pjrt` XLA FFI adapter, and re-executes it
after serializing and reloading the enclosing executable. It runs when `RYFT_PJRT_RUN_CUTILE_SEAM_PROBE=1` and the
`cuda-13` feature are set.

The Mosaic GPU probes (`src/jax/mosaic_gpu.rs`) build a vector-add and a tiled-matmul kernel through typed `ryft-mlir`
constructors following the pinned JAX host ABI, serialize them with the `mosaic_gpu-serde` pass, and embed the
bytecode in a StableHLO `custom_call @mosaic_gpu_v2`. Module construction, serialization, binary round trips, and
program construction are tested portably; compilation, execution, AOT reload, PTX dump evidence
(`MOSAIC_GPU_DUMP_TO` and `MOSAIC_GPU_DUMP_PTX`), and exact runtime diagnostics run when
`RYFT_PJRT_RUN_MOSAIC_GPU_SEAM_PROBE=1` and a `cuda-12` or `cuda-13` feature are set. The
`pallas_gpu_seam_probes.yaml` workflow runs both slices on Linux x86_64 NVIDIA runners, including a
`compute-sanitizer --tool memcheck --leak-check full` pass; no aarch64 GPU runner is available, so aarch64 stays
build-only.

- [ ] For JAX-level support we need to be able to load the MLIR dialects and passes that are listed
  [here](https://github.com/jax-ml/jax/blob/d13a4754e3a8e265008ac3ab23c27d4cb244b8b9/jax/_src/interpreters/mlir.py#L601).
- [ ] We want to be able to instantiate a model (potentially with a sharding config) doing all necessary allocations.
  Then, we also want to be able to run initializers for the model parameters or load from files, making sure that
  only the relevant/appropriate shard is loaded on each device.
- [ ] The CUDA PJRT/JAX plugin does some additional initialization:
  ```python
  if cuda_plugin_extension:
    xla_client.register_custom_call_handler(
        "CUDA",
        functools.partial(
            cuda_plugin_extension.register_custom_call_target, c_api
        ),
    )
    for _name, _value in cuda_plugin_extension.ffi_registrations().items():
      xla_client.register_custom_call_target(
          _name, _value, platform='CUDA', api_version=1
      )
    xla_client.register_custom_type_id_handler(
        "CUDA",
        functools.partial(
            cuda_plugin_extension.register_custom_type_id, c_api
        ),
    )
    triton.register_compilation_handler(
        "CUDA",
        functools.partial(
            cuda_plugin_extension.compile_triton_to_asm, c_api
        ),
    )
  ```
