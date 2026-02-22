# **Ryft MLIR:** Rust Bindings for MLIR

This crate provides high-level Rust bindings for [MLIR](https://mlir.llvm.org/) and a growing set of MLIR dialects
used by XLA-related tooling (including [StableHLO](https://openxla.org/stablehlo) and Shardy). It wraps the MLIR C API
exposed by `ryft-xla-sys` with ownership-aware Rust types for contexts, modules, attributes, types, values, operations,
regions/blocks, pass managers, and the execution engine. `ryft-mlir` is intended to make building and transforming MLIR
programs ergonomic in Rust.

This crate currently does not define crate-specific feature flags. Native artifact and toolchain configuration is
handled by `ryft-xla-sys`. For more information on artifact loading and platform support, refer to
[`crates/ryft-xla-sys/README.md`](../ryft-xla-sys/README.md).

## Introduction

The following is an example for how you can create an MLIR context, load the `func` and `stablehlo` dialects,
parse a StableHLO module, and verify it:

```rust
use ryft::mlir::*;

fn main() {
    let context = Context::new();
    context.load_dialect(DialectHandle::func());
    context.load_dialect(DialectHandle::stable_hlo());
    let module = context
        .parse_module(
            r#"
            module {
              func.func @main(%lhs: tensor<2x1xi32>, %rhs: tensor<2x1xi32>) -> tensor<2x1xi32> {
                %0 = stablehlo.add %lhs, %rhs : tensor<2x1xi32>
                return %0 : tensor<2x1xi32>
              }
            }
            "#,
        )
        .unwrap();
    assert!(module.verify());
    println!("{module}");
}
```

At a high level, a typical workflow for working with `ryft-mlir` looks as follows:

1. **Create a Context:** Construct a `Context` (or `Context::new_with_registry(..)` if you need explicit dialect
   registration behavior) and configure threading as needed.
2. **Load/Register Dialects:** Use `DialectHandle`/`DialectRegistry` APIs to make dialects available before IR
   construction or parsing.
3. **Build or Parse IR:** Build modules/operations programmatically via typed APIs (e.g., dialect-specific constructors)
   or parse textual MLIR via `context.parse_module(..)` and `context.parse_operation(..)`.
4. **Transform and Validate:** Run pass pipelines using `PassManager`, then validate with operation/module verification.
5. **Lower and/or Execute:** Lower to target dialects and optionally use `ExecutionEngine`, or hand StableHLO programs
   to [`ryft-pjrt`](../ryft-pjrt) for PJRT compilation and execution.

The following is an example showing how to run optimization passes on an MLIR module:

```rust
use ryft::mlir::*;

fn main() {
    let context = Context::new();
    context.load_dialect(DialectHandle::arith());
    context.load_dialect(DialectHandle::func());
    let module = context
        .parse_module(
            r#"
            module {
              func.func @main() -> i32 {
                %0 = arith.constant 7 : i32
                %1 = arith.constant 7 : i32
                %2 = arith.addi %0, %1 : i32
                return %2 : i32
              }
            }
            "#,
        )
        .unwrap();
    let mut pass_manager = context.pass_manager();
    pass_manager.add_pass(ryft::mlir::dialects::builtin::passes::create_transforms_cse_pass());
    pass_manager.add_pass(ryft::mlir::dialects::builtin::passes::create_transforms_canonicalizer_pass());
    assert!(pass_manager.run(&module.as_operation()).is_success());
}
```

## Roadmap / TODOs

- [ ] Rename `as_type_ref`, `as_affine_expression_ref`, etc., to just `as_ref`, maybe?
- [ ] Remove uses of `.expect` and `panic!` (and `.unwrap` where it makes sense), and rely on error propagation
  instead.
- [ ] `BooleanAttributeRef::is<IntegerAttributeRef>` panics (and the same for a 1-bit integer attribute in reverse).
- [ ] Add `Context` constructors like `i32_type`, etc. Maybe also `bool_type` as an alias for `i1_type`?
- [ ] Clean up the API we have around elements attributes and make it better typed if possible.
- [ ] Figure out whether we can use mutable references for the context in more places.
- [ ] Add wrappers for `mlirOperationReplaceUsesOfWith` and `mlirBlockArgumentSetLocation`.
- Support more MLIR dialects:
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
    - [/] `shardy`: Current support for the Shardy dialect is only partial and aimed at attributes and operations that
      are relevant when building StableHLO programs to be compiled by XLA, and not covering attributes and operations
      that are internal to the Shardy compiler. Also, the current support was added hastily and is likely to change as
      we start using these attributes and operations in practice.
    - [ ] `sparse_tensor`
    - [x] `stable_hlo`
        - [ ] For `stable_hlo::bitcast_convert` can we have the constructor only require the output data type and infer
          the output shape automatically?
        - [ ] For `stable_hlo::dynamic_broadcast` we should be able to use `known_expanding_dimensions` and
          `known_non_expanding_dimensions` to refine the inferred output shape. Also, we may be able to leverage the
          fact that certain dimensions are size 1 or not.
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
