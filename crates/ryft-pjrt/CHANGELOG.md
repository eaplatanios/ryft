# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added shared `ExecutionFence` and `Execution<Output>` wrappers for explicit whole-execution readiness and
  asynchronous error observation. `LoadedExecutable::execute` now returns an `Execution<Vec<ExecutionDeviceOutputs>>`
  whose fence joins the per-device completion events of the launch, and `ExecutionDeviceOutputs` no longer exposes a
  per-device `done` event.
- Added support for the new `PJRT_Buffer_Bitcast` C API function.
- Added support for the new `PJRT_Device_ClearMemoryStats` C API function.
- Added support for the new `PJRT_Error_ForEachPayload` C API function and for providing payload-aware safe Rust
  wrappers for error buffers and execution poisoning.
- Added support for the new `PJRT_TopologyDescription_Fingerprint` C API function.
- Added support for the new `PJRT_Executable_ParameterMemoryKinds` C API function.
- Added support for the new `PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace` C API function.
- Added support for the new `PJRT_TopologyDescription_GetMemorySpaceKindIds` C API function.
- Added support for the new `PJRT_LoadOptions`.
- Added support for the new `PJRT_HostMemoryAllocator` extension and its owned host-memory allocation wrapper.
- Added support for the new `PJRT_Xla_Transform` extension through a safe `XlaTransform` trait API.
- Added the `mps` feature and `load_mps_plugin()` for loading the `jax-mps` PJRT plugin.
- Added `BufferType::element_size_in_bytes`.
- Added safe wrappers for OpenXLA's XProf-to-feedback-directed-profile conversion and deterministic multi-profile
  aggregation used by profile-guided latency estimation.

### Changed

- Made PJRT events, execution fences, executions, and buffers thread-safe through shared ownership and narrow native
  handle wrappers that reflect PJRT's thread-safety contracts. Event callbacks now require `Send + 'static`, the
  unsafe `EventHandle` was replaced by the safe shared-ownership `EventPromise`, and asynchronous host-buffer
  ownership uses `Arc` instead of `Rc<RefCell<_>>`. `Client::event` and `Plugin::event` now return an
  `(Event, EventPromise)` pair in which the non-clonable `EventPromise` is the only way to set/trigger the event and
  is consumed by `set`, structurally enforcing the PJRT C API's `PJRT_Event_Set` contract (only events created through
  `PJRT_Event_Create` may be set, at most once). `Event` is `Send` but deliberately not `Sync` because the PJRT C API
  does not guarantee that overlapping consumer calls are thread-safe, and dropping a pending `Event` releases the task
  waker registered through its `Future` implementation. `Client::borrowed_mut_buffer` is now explicitly `unsafe` because
  callers must prevent safe access to the shared host allocation while PJRT may mutate it. Buffer external
  reference-count operations now document their exceptional external-synchronization requirement.
- Changed host-buffer and host-to-device-transfer keep-alive closures to be leaked instead of released when a failure
  occurs after the data pointer has been handed to a successful PJRT call, so that the shared host data can never be
  freed while an in-flight transfer may still be reading it.
- Required `KeyValueStore` implementations to be `Send + Sync` because PJRT may invoke their callbacks concurrently.
  `Client` now obtains its thread-safety structurally through a narrow native-handle wrapper instead of whole-type
  unsafe `Send` and `Sync` implementations.
- Updated our PJRT C API bindings for version `0.111`.
- Changed `Buffer::copy_to_host` to fall back to the buffer's reported on-device byte size when a PJRT plugin
  returns a successful host-copy size query without populating `dst_size`.
- Changed `BufferSpecification` to carry a concrete `Layout`, materializing dense defaults during construction and
  parsing before values cross layout-sensitive PJRT C API calls.
- Expanded executable compiled-memory statistics support to include total allocator bytes, indefinite allocations,
  and peak unpadded heap bytes.
- Changed `TiledLayout::minor_to_major` to `Vec<u64>` from `Vec<i64>`.
- Changed `ExecutionInput::buffer` to an `Arc<Buffer<'o>>` instead of a `Buffer<'o>`.
- Changed `Memory` equality to fall back to memory-kind strings when a PJRT plugin does not implement memory kind IDs.

## [0.0.2] - 2026-03-02

### Added

- Added support for `BufferType::S1` and `BufferType::U1`.
- Added support for the new `PJRT_Device_GetAttributes` C API function.
- Added support for the new `PJRT_Client_Load` C API function.
- Added support for the new `PJRT_LoadedExecutable_AddressableDeviceLogicalIds` C API function.

### Changed

- Updated our PJRT C API bindings for version `0.97`.
- Updated the layouts extension bindings to version `4` and added support for executable parameter layout queries.
- Updated the FFI extension bindings to version `3` and added support for setting and getting the execution context
  for specific execution stages.

## [0.0.1] - 2026-02-22

### Added

- Initial release.

<!-- next-url -->
[0.0.1]: https://github.com/eaplatanios/ryft/compare/v0.0.1...HEAD
