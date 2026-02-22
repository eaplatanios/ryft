# **Ryft PJRT:** Rust Bindings for PJRT

This crate provides high-level Rust bindings for [PJRT](https://openxla.org/xla/pjrt), the runtime interface used by
XLA backends (e.g., CPU, GPU, TPU, etc.). It wraps the PJRT C API exposed by `ryft-xla-sys` with ownership-aware Rust
types for PJRT plugins, clients, devices, memory spaces, buffers, program compilation and execution, distributed
runtime coordination, and various PJRT extensions (e.g., layouts, shardings, FFI, profiler, etc.). `ryft-pjrt` is the
primary crate to use when writing Rust code against PJRT in this repository. It is built on top of `ryft-xla-sys`,
which contains the low-level bindings and native artifact building and/or downloading.

Note that this crate forwards the following feature flags directly to `ryft-xla-sys`: `cuda-12`, `cuda-13`, `rocm-7`,
`tpu`, `neuron`, and `metal`. For information on what those flags represent and guidance on how to use them, refer to
[`crates/ryft-xla-sys/README.md`](../ryft-xla-sys/README.md).

## Introduction

The following is an example for how you can load the built-in CPU PJRT plugin, create a PJRT client for it, and then
log information about all devices that are _addressable_ by that client:

```rust
use ryft_pjrt::{load_cpu_plugin, ClientOptions};

fn main() {
    let plugin = load_cpu_plugin()?;
    let client = plugin.client(ClientOptions::default())?;
    println!("Platform Name: {}", client.platform_name()?);
    println!("Platform version: {}", client.platform_version()?);
    println!("PJRT API Version: {}", client.version());
    println!("Devices:");
    for device in client.addressable_devices()? {
        println!("  - Device {} ({})", device.id()?, device.kind()?);
    }
}
```

At a high-level, a typical workflow for working with `ryft-pjrt` looks as follows:

1. **Load a Plugin:** Load the built-in CPU plugin using `load_cpu_plugin()` or, depending on the feature flags you have
   enabled, accelerator-specific plugins using `load_cuda_12_plugin()`, `load_cuda_13_plugin()`, `load_rocm_7_plugin()`,
   `load_tpu_plugin()`, `load_neuron_plugin()`, and `load_metal_plugin()`. You can also load a third-party plugin by
   using `load_plugin(path)` and providing a path to the relevant shared library.
2. **Create a Client:** Create a single-host client using `plugin.client(options)` and a multi-host client using
   `plugin.client_with_key_value_store(options, store)`. In order to use the latter properly, you will also need
   to leverage `plugin.distributed_runtime_service(..)`, `plugin.distributed_runtime_client(..)`,
   `DistributedKeyValueStore::new(..)`, and `client.update_global_process_information(..)`.
3. **Manage Buffers:** Create buffers using multiple APIs like `client.buffer(..)`, `client.borrowed_buffer(..)`,
   `client.borrowed_mut_buffer(..)`, etc. There are also multiple APIs for managing data transfers between memory
   spaces, between devices, and between hosts.
4. **Compile & Execute Programs:** Use `client.compile(..)` to compile [StableHLO](https://openxla.org/stablehlo) and
   HLO programs that you can then execute with `executable.execute(..)`. You can also use the
   [`ryft-mlir`](../ryft-mlir) crate for safe Rust wrappers for building [MLIR](https://mlir.llvm.org/) programs
   (including support for the StableHLO dialect). Note that execution in PJRT is asynchronous.

The following is an example that shows how you can compile and execute a simple StableHLO program:

```rust
use ryft_pjrt::protos::*;
use ryft_pjrt::*;

fn main() {
    let plugin = load_cpu_plugin()?;
    let client = plugin.client(ClientOptions::default())?;
    let program = Program::Mlir {
        bytecode: br#"
            module {
              func.func @main(%arg0: tensor<2x1xi32>, %arg1: tensor<2x1xi32>) -> tensor<2x1xi32> {
                %0 = stablehlo.add %arg0, %arg1 : tensor<2x1xi32>
                return %0 : tensor<2x1xi32>
              }
            }
        "#
            .to_vec(),
    };
    let options = CompilationOptions {
        executable_build_options: Some(ExecutableCompilationOptions {
            device_ordinal: -1,
            replica_count: 1,
            partition_count: 1,
            ..Default::default()
        }),
        matrix_unit_operand_precision: Precision::Default as i32,
        ..Default::default()
    };
    let executable = client.compile(&program, &options)?;
    let device = executable.addressable_devices()?[0].clone();

    let lhs = [7i32.to_ne_bytes(), (-1i32).to_ne_bytes()].concat();
    let rhs = [35i32.to_ne_bytes(), (-41i32).to_ne_bytes()].concat();
    let lhs_buffer = client.buffer(&lhs, BufferType::I32, &[2, 1], None, device.clone(), None)?;
    let rhs_buffer = client.buffer(&rhs, BufferType::I32, &[2, 1], None, device, None)?;
    let inputs = ExecutionDeviceInputs {
        inputs: &[
            ExecutionInput { buffer: lhs_buffer, donatable: false },
            ExecutionInput { buffer: rhs_buffer, donatable: false },
        ],
        ..Default::default()
    };

    let mut outputs = executable.execute(vec![inputs], 0, None, None, None, None)?;
    let mut outputs = outputs.remove(0);
    outputs.done.r#await()?;

    let result = outputs.outputs.remove(0).copy_to_host(None)?.r#await()?;
    println!("{result:?}");
}
```

## Roadmap / TODOs

- [ ] Cache extension lookups on first access so `Api::*_extension` methods avoid repeated linked-list traversal.
  The importance of this TODO item depends heavily on how extensively we need to rely on repeated invocations of
  PJRT extension functions, since each of those calls will require a traversal of that linked list.
- [ ] Make the `compute_capability` attribute first class for PJRT GPU plugins and update `triton.rs` to use it.
- Support all known PJRT extensions:
    - CPU Plugin:
        - [x] FFI
        - [x] Layouts
        - [x] Memory descriptions
        - [x] Shardings
    - GPU Plugin:
        - [x] Cross-Host Transfers
        - [x] GPU Custom Call
        - [ ] Custom Partitioner
        - [x] FFI
        - [x] Layouts
        - [x] Memory Descriptions
        - [x] Profiler
        - [x] Shardings
        - [x] Stream
        - [x] Triton
    - TPU Plugin:
        - [ ] Callbacks
        - [ ] TPU Topology
        - [ ] TPU Executable
        - [ ] Megascale
    - Extensions that do not appear to be implemented for any PJRT plugins provided by Google:
        - [x] Executable Metadata
        - [x] Host Allocator
        - [ ] Phase Compile

#### License

<sup>
Licensed under either <a href="../../LICENSE-APACHE">Apache License, Version 2.0</a>
or <a href="../../LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this crate by you,
as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
</sub>
