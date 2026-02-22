# Ryft MLIR

## TODOs

- [ ] Rename `as_type_ref`, `as_affine_expression_ref`, etc., to just `as_ref`, maybe?
- [ ] Remove uses of `.expect` and `panic!` (and `.unwrap` where it makes sense), and rely on error propagation instead.
- [ ] `BooleanAttributeRef::is<IntegerAttributeRef>` panics (and the same for a 1-bit integer attribute in reverse).
- [ ] Add `Context` constructors like `i32_type`, etc. Maybe also `bool_type` as an alias for `i1_type`?
- [ ] Clean up the API we have around elements attributes and make it better typed if possible.
- [ ] Figure out whether we can use mutable reference for the context in more places.
- [ ] `mlirOperationReplaceUsesOfWith` and `mlirBlockArgumentSetLocation`.

## Dialect Support

- [/] `affine`
    - [ ] Add support for operations.
- [/] `arith`
    - [ ] Add support for operations: `addui_extended`, `mulsi_extended`, `mului_extended`.
    - [ ] Add support for attributes: `FastMathFlags`, `IntegerOverflowFlags`, `RoundingMode`, `AtomicRMWKind`.
- [ ] `async`
- [x] `builtin`
- [x] `cf`
- [ ] `chlo`
- [ ] `emit_c`
    - [ ] Add support for types: `EmitCArrayType`, `EmitCLValueType`, `EmitCOpaqueType`, `EmitCPointerType`,
      `EmitCPtrDiffTType`, `EmitCSignedSizeTType`, `EmitCSizeTType`.
    - [ ] Add support for operations.
- [x] `func`
- [ ] `gpu`
    - [ ] Add support for types: `GPUAsyncTokenType`.
    - [ ] Add support for operations.
- [x] `index`
- [ ] `linalg`
- [ ] `llvm`
    - [ ] Add support for types: `LLVMPointerType`, `LLVMStructType`.
    - [ ] Add support for operations.
- [ ] `memref`
- [ ] `mhlo`
- [ ] [`mosaic_gpu`](https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/gpu/mosaic_gpu.td)
- [ ] [`mosaic_tpu`](https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/tpu/tpu.td)
- [ ] `nvgpu`
    - [ ] Add support for types: `NVGPUTensorMapDescriptorType`.
    - [ ] Add support for operations.
- [ ] `pdl`
    - [ ] Add support for types: `PDLAttributeType`, `PDLOperationType`, `PDLRangeType`, `PDLType`,
      `PDLTypeType`, `PDLValueType`.
    - [ ] Add support for operations.
- [x] `quant`
    - [ ] Add support for types: `QuantizedType`, `AnyQuantizedType`, `CalibratedQuantizedType`,
      `UniformQuantizedPerAxisType`, `UniformQuantizedType`, `UniformQuantizedSubChannelType`.
    - [ ] Add support for operations.
- [ ] `scf`
- [ ] `shape`
- [/] `shardy`: Current support for the Shardy dialect is only partial and aimed at attributes and operations that are
  relevant when building StableHLO programs to be compiled by XLA, and not covering attributes and operations that are
  internal to the Shardy compiler. Also, the current support was added hastily and is likely to change as we start using
  these attributes and operations in practice.
- [ ] `sparse_tensor`
- [x] `stable_hlo`
    - [ ] For `stable_hlo::bitcast_convert` can we have the constructor only require the output data type and infer
      the output shape automatically?
    - [ ] For `stable_hlo::dynamic_broadcast` we should be able to use `known_expanding_dimensions` and
      `known_non_expanding_dimensions` to refine the inferred output shape. Also, we may be able to leverage the fact
      that certain dimensions are size 1 or not.
    - [ ] For `stable_hlo::dynamic_gather` can we infer the output shape automatically? Or at least part of it?
    - [ ] Add setters for StableHLO operation attributes (and more broadly perhaps).
        - Should we maybe add a `StableHloOperation` trait and then implement Shardy setters for that trait?
    - [ ] Add checks for operation arguments when constructing them to prevent panics and return informative errors:
        - https://openxla.org/stablehlo/spec
        - https://github.com/openxla/stablehlo/blob/4c0d4841519aed22e3689c30b72a0e4228051249/stablehlo/dialect/StablehloOps.cpp
- [ ] `tensor`
- [ ] `triton`
- [ ] `transform`
    - [ ] Add support for types: `TransformAnyOpType`, `TransformAnyParamType`, `TransformAnyValueType`,
      `TransformOperationType`, `TransformParamType`.
    - [ ] Add support for operations.
- [ ] `versioned_hlo`

## License

Licensed under either of:

- Apache License, Version 2.0, ([LICENSE-APACHE](../../LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../../LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you,
as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
