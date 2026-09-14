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
`pallas_gpu_seam_probes.yaml` workflow runs both slices on Linux x86_64 NVIDIA runners. CI has no aarch64 GPU runner;
the DGX Spark provides manual aarch64 CUDA 13 execution qualification. Neither substitutes for the other platform
builds. The remaining platform matrix is deferred during kernel development under the approved local qualification
exception in `plan-pallas.md`.

The workflow builds the CUDA plugin from the current source and uses the same plugin for probes and an ordinary
PJRT compilation control. CUDA memcheck with full leak checking retains complete logs; `tools/compare_sanitizer_leaks.py`
requires identical cuDNN initialization allocations in the control and probe, with no other sanitizer errors. Matching
retained allocations do not establish a leak-free process. Racecheck, synccheck, and initcheck run separately, and the
generic CUDA launcher suite covers its concurrent cache contract. These CUDA tools do not establish host ASan/TSan
coverage. PTX evidence must identify each slice's entry and device architecture together.

The cuTile exporter requires `cuda-tile[tileiras]==1.5.0` for the `cutile_python_v2` calling convention. Verification
executes the exported cubin directly through the Ryft probe; recompiling its Python source through JAX would not
verify that artifact. Worker failures and timeouts retain diagnostics without replacing previously exported files.

The macro-authored Mosaic experiments in `src/kernels.rs` retain a separate source path from those handwritten ABI
probes. They declare vector addition, a scalar sum reduction, and tiled matrix multiplication with the portable
`#[kernel]` macro. The shared cases use partial tiles and independent scalar oracles; the core reference interpreter
checks the same definitions before native execution. The GPU runner uses the generated `definition`, ordinary XLA
tracing with `stage_kernel`, and an explicit Mosaic compiler binding. Each native case has an independent test, so a
failure does not hide the other cases. Adapter errors fail the test; GPU requests do not select a host interpreter as
a replacement backend.

An additional explicit-builder case copies 67 elements from an input reference into aligned scratch storage, waits
for the copy, and publishes the result. This qualifies the adapter's asynchronous memory operations independently
of the macro surface; it does not imply that the macro accepts asynchronous-copy syntax.

Run the portable oracle cases with `cargo test -p ryft-experimental kernels::test_macro_kernels_interpretation`.
After building the CUDA plugin, run native cases with
`RYFT_PJRT_RUN_MOSAIC_GPU_KERNELS=1 cargo test -p ryft-experimental --features mosaic-gpu,cuda-13 kernels::gpu`.
This native gate is distinct from `RYFT_PJRT_RUN_MOSAIC_GPU_SEAM_PROBE`, so successful handwritten ABI probes cannot
stand in for macro-to-adapter qualification. The new reduction syntax `input.load().sum([0])` emits a canonical
reduction operation; its axes must be integer literals, and ordinary type inference validates their bounds.

The ignored assertion-failure test intentionally traps the device and must run in a separate process after the positive
cases. It requires successful native compilation followed by a CUDA launch or illegal-instruction failure; the failed
device context is then discarded.

```sh
RYFT_PJRT_RUN_MOSAIC_GPU_ASSERTION_FAILURE=1 cargo test -p ryft-experimental \
  --features mosaic-gpu,cuda-13 kernels::gpu::test_assertion_failure_on_cuda -- --exact --ignored
```

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
