# TODOs

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
