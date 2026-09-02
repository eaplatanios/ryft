//! Test-only Mosaic GPU vertical slices.
//!
//! The kernels in this module are built through typed `ryft-mlir` constructors and mirror the host ABI that the pinned
//! JAX `jax.experimental.mosaic.gpu` lowering emits (`core.py::_lower_as_gpu_kernel`, `launch_context.py::_launch`,
//! and `utils.py::ptr_as_memref`): a `func.func @<kernel>_mosaic_gpu(!llvm.ptr, !llvm.ptr)` entry with
//! `llvm.emit_c_interface`, `mosaic_gpu.arch_major`/`mosaic_gpu.arch_minor` module attributes, a `global_scratch`
//! constant-memory global, a private `mosaic_gpu_init_tma_desc` declaration, buffer pointers loaded from the second
//! argument into strided memref descriptors, and an asynchronous `gpu.launch` body. The serialized module is consumed
//! by the upstream `mosaic_gpu_v2` XLA FFI target through a StableHLO `custom_call` whose `backend_config` carries the
//! SHA-256 `kernel_hash`, the `mosaic_gpu-serde{serialize=true}` bytecode, and the `use_custom_barrier` and
//! `uses_xla_collective_metadata` flags (`jaxlib/mosaic/gpu/custom_call.cc`).

use std::path::Path;
use std::{env, fs};

use ryft_mlir::dialects::gpu::{Dim3, LaunchProperties};
use ryft_mlir::dialects::llvm::Linkage;
use ryft_mlir::dialects::mosaic::gpu::{mosaic_gpu_serde_pass_manager, mosaic_gpu_serde_version};
use ryft_mlir::dialects::stable_hlo::{
    CustomCallApiVersion, CustomCallMemoryLayouts, CustomCallOperation, DetachedCustomCallOperation,
    OutputOperandAliasAttributeRef,
};
use ryft_mlir::dialects::{arith, builtin, func, gpu, llvm, memref, scf, stable_hlo};
use ryft_mlir::{
    Attribute, Block, Context, DetachedBlock, DetachedOp, DialectHandle, Error, Module, Operation, Size, StringRef,
    SymbolVisibility, Type, TypeRef, UnknownLocationRef, Value, ValueRef,
};
use ryft_xla_sys::mlir::dialects::mosaic::gpu::{MOSAIC_GPU_FFI_TARGET, MOSAIC_GPU_SERDE_VERSION};
use sha2::{Digest, Sha256};

use crate::tests::{TestPlatform, test_compilation_options, test_for_each_platform};
use crate::{
    BufferType, Client, Device, ExecutionDeviceInputs, ExecutionInput, LoadOptions, Program, Value as PjrtValue,
};

/// Attribute the pinned JAX lowering attaches to each kernel-argument memref cast
/// (`launch_context.KERNEL_ARG_ID_ATTR`).
const KERNEL_ARGUMENT_INDEX_ATTRIBUTE: &str = "mosaic_gpu.from_kernel_arg_idx";

/// Attribute marking the original kernel-argument memref casts (`launch_context.ORIGINAL_KERNEL_ARG_ATTR`).
const ORIGINAL_KERNEL_ARGUMENT_ATTRIBUTE: &str = "mosaic_gpu.original_kernel_arg";

/// Compute capability recorded in the portable module fixtures (Hopper), matching JAX's export default.
const PORTABLE_COMPUTE_CAPABILITY: (u32, u32) = (9, 0);

const VECTOR_ADD_LENGTH: usize = 1024;
const VECTOR_ADD_BLOCK: usize = 256;
const MATMUL_M: usize = 32;
const MATMUL_K: usize = 8;
const MATMUL_N: usize = 48;
const MATMUL_TILE: usize = 16;

/// Values that live as long as the MLIR context, which is the lifetime `DetachedBlock` hands out for its arguments
/// and appended operation results.
type KernelValue<'c, 't> = ValueRef<'c, 'c, 't>;

/// Appends `operation` to `block` and returns its first result.
fn append<'c, 't, O: DetachedOp<'c, 'c, 't>>(
    block: &mut DetachedBlock<'c, 't>,
    operation: O,
) -> Result<KernelValue<'c, 't>, Error> {
    Ok(block.append_operation(operation)?.result(0)?.as_ref())
}

/// Loads kernel argument `index` from the `buffers` pointer array and wraps it in a row-major `memref<...xf32>`
/// descriptor exactly like `utils.getelementptr` followed by `utils.ptr_as_memref` in the pinned JAX lowering.
fn kernel_argument<'c, 't>(
    context: &'c Context<'t>,
    block: &mut DetachedBlock<'c, 't>,
    buffers: KernelValue<'c, 't>,
    index: usize,
    shape: &[usize],
    location: UnknownLocationRef<'c, 't>,
) -> Result<KernelValue<'c, 't>, Error> {
    let pointer_type = context.llvm_pointer_type(0)?.as_ref();
    let i32_type = context.signless_integer_type(32);
    let i64_type = context.signless_integer_type(64);
    let slot = append(
        block,
        llvm::get_element_ptr(
            buffers,
            &[],
            pointer_type,
            context.dense_i32_array_attribute(&[index as i32])?.as_ref(),
            context.type_attribute(pointer_type).as_ref(),
            None,
            location,
        )?,
    )?;
    let pointer = append(block, llvm::load(slot, pointer_type, None, false, location)?)?;

    // The strided memref descriptor is `(allocated, aligned, offset, sizes, strides)` with a zero offset.
    let array_type = context.llvm_array_type(i64_type, shape.len() as u64)?.as_ref();
    let descriptor_type = context
        .llvm_literal_struct_type(&[pointer_type, pointer_type, i64_type.as_ref(), array_type, array_type], false)?
        .as_ref();
    let insert = |block: &mut DetachedBlock<'c, 't>, descriptor, value, position: &[i64]| {
        let position = context.dense_i64_array_attribute(position)?.as_ref();
        append(block, llvm::insert_value(descriptor, value, descriptor_type, position, location)?)
    };
    let i64_constant = |block: &mut DetachedBlock<'c, 't>, value: usize| {
        append(block, llvm::constant(context.integer_attribute(i64_type, value as i64), i64_type, location)?)
    };
    let mut descriptor = append(block, llvm::undef(descriptor_type, location)?)?;
    descriptor = insert(block, descriptor, pointer, &[0])?;
    descriptor = insert(block, descriptor, pointer, &[1])?;
    let offset = i64_constant(block, 0)?;
    descriptor = insert(block, descriptor, offset, &[2])?;
    for (dimension, size) in shape.iter().enumerate() {
        let size = i64_constant(block, *size)?;
        descriptor = insert(block, descriptor, size, &[3, dimension as i64])?;
    }
    for dimension in 0..shape.len() {
        let stride = i64_constant(block, shape[dimension + 1..].iter().product())?;
        descriptor = insert(block, descriptor, stride, &[4, dimension as i64])?;
    }

    let shape = shape.iter().map(|size| Size::Static(*size)).collect::<Vec<_>>();
    let memref_type = context.mem_ref_type(context.float32_type(), shape.as_slice(), None, None, location)?;
    let mut cast = builtin::unrealized_conversion_cast(&[descriptor], &[memref_type.as_ref()], location)?;
    cast.set_attribute(KERNEL_ARGUMENT_INDEX_ATTRIBUTE, context.integer_attribute(i32_type, index as i64));
    cast.set_attribute(ORIGINAL_KERNEL_ARGUMENT_ATTRIBUTE, context.unit_attribute());
    append(block, cast)
}

/// Builds a Mosaic GPU module following the pinned JAX host ABI.
///
/// `body` receives the `gpu.launch` body block, the kernel-argument memrefs in `argument_shapes` order, and the twelve
/// launch block arguments (`block_id.{x,y,z}`, `thread_id.{x,y,z}`, `grid_dim.{x,y,z}`, `block_dim.{x,y,z}`). The
/// launch body is terminated by this function.
fn mosaic_gpu_module<'c, 't, B>(
    context: &'c Context<'t>,
    kernel_name: &str,
    compute_capability: (u32, u32),
    argument_shapes: &[&[usize]],
    grid: [usize; 3],
    block: [usize; 3],
    body: B,
) -> Result<Module<'c, 't>, Error>
where
    B: FnOnce(&mut DetachedBlock<'c, 't>, &[KernelValue<'c, 't>], &[KernelValue<'c, 't>]) -> Result<(), Error>,
{
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    let location = context.unknown_location();
    let module = context.module(location)?;
    let i32_type = context.signless_integer_type(32);
    let index_type = context.index_type();
    let pointer_type = context.llvm_pointer_type(0)?.as_ref();
    let i64_type = context.signless_integer_type(64).as_ref();

    let mut module_operation = module.as_operation()?;
    module_operation.set_attribute("sym_name", context.string_attribute(kernel_name));
    module_operation
        .set_attribute("mosaic_gpu.arch_major", context.integer_attribute(i32_type, compute_capability.0 as i64));
    module_operation
        .set_attribute("mosaic_gpu.arch_minor", context.integer_attribute(i32_type, compute_capability.1 as i64));

    // Runtime declarations shared by every Mosaic GPU kernel: the TMA descriptor initializer and the constant-memory
    // scratch global whose size the runtime fills in.
    let init_tma_desc_arguments =
        [pointer_type, pointer_type, i64_type, i64_type, pointer_type, pointer_type, i64_type, pointer_type];
    module.body()?.append_operation(func::func(
        "mosaic_gpu_init_tma_desc",
        func::FuncAttributes {
            arguments: init_tma_desc_arguments.iter().copied().map(Into::into).collect(),
            visibility: SymbolVisibility::Private,
            ..Default::default()
        },
        context.region(),
        location,
    )?)?;
    let scratch_type = context.llvm_array_type(context.signless_integer_type(8), 0)?;
    module.body()?.append_operation(llvm::global(
        context.type_attribute(scratch_type).as_ref(),
        false,
        context.string_attribute("global_scratch").as_ref(),
        context.llvm_linkage_attribute(Linkage::External)?.as_ref(),
        false,
        false,
        false,
        None,
        None,
        Some(context.integer_attribute(i32_type, 4).as_ref()),
        None,
        None,
        None,
        None,
        None,
        None,
        context.region(),
        location,
    )?)?;

    // Host entry point: the first argument is the XLA stream token and the second is the kernel-argument pointer array.
    let mut function_block = context.block(&[(pointer_type, location), (pointer_type, location)]);
    let token_pointer = function_block.argument(0)?.as_ref();
    let buffers = function_block.argument(1)?.as_ref();
    let token_type = context.gpu_async_token_type()?.as_ref();
    let token =
        append(&mut function_block, builtin::unrealized_conversion_cast(&[token_pointer], &[token_type], location)?)?;
    let mut memrefs = Vec::with_capacity(argument_shapes.len());
    for (index, shape) in argument_shapes.iter().enumerate() {
        memrefs.push(kernel_argument(context, &mut function_block, buffers, index, shape, location)?);
    }
    let mut dimensions = Vec::with_capacity(6);
    for size in grid.into_iter().chain(block) {
        let size = context.integer_attribute(index_type, size as i64);
        dimensions.push(append(&mut function_block, arith::constant(size, location)?)?);
    }
    let dynamic_shared_memory_size =
        append(&mut function_block, arith::constant(context.integer_attribute(i32_type, 0), location)?)?;

    let mut launch_block = context.block(&vec![(index_type, location); 12]);
    let launch_arguments = (0..12)
        .map(|index| launch_block.argument(index).map(|argument| argument.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    body(&mut launch_block, memrefs.as_slice(), launch_arguments.as_slice())?;
    launch_block.append_operation(gpu::terminator(location)?)?;
    function_block.append_operation(gpu::launch(
        LaunchProperties {
            async_dependencies: vec![token],
            grid_size: Dim3 { x: dimensions[0], y: dimensions[1], z: dimensions[2] },
            block_size: Dim3 { x: dimensions[3], y: dimensions[4], z: dimensions[5] },
            cluster_size: None,
            dynamic_shared_memory_size: Some(dynamic_shared_memory_size),
            module: None,
            function: None,
            workgroup_attributions: None,
            is_async: true,
        },
        launch_block.try_into()?,
        location,
    )?)?;
    function_block.append_operation(func::r#return(&[] as &[KernelValue<'c, 't>], location)?)?;
    module.body()?.append_operation(func::func(
        format!("{kernel_name}_mosaic_gpu").as_str(),
        func::FuncAttributes {
            arguments: vec![pointer_type.into(), pointer_type.into()],
            llvm_emit_c_interface: true,
            ..Default::default()
        },
        function_block.try_into()?,
        location,
    )?)?;
    Ok(module)
}

/// Builds `out[i] = lhs[i] + rhs[i]` over `f32[VECTOR_ADD_LENGTH]` with one thread per element.
fn vector_add_module<'c, 't>(
    context: &'c Context<'t>,
    compute_capability: (u32, u32),
) -> Result<Module<'c, 't>, Error> {
    let location = context.unknown_location();
    let f32_type = context.float32_type();
    let shape: &[usize] = &[VECTOR_ADD_LENGTH];
    mosaic_gpu_module(
        context,
        "vector_add",
        compute_capability,
        &[shape, shape, shape],
        [VECTOR_ADD_LENGTH / VECTOR_ADD_BLOCK, 1, 1],
        [VECTOR_ADD_BLOCK, 1, 1],
        |block, memrefs, launch| {
            let offset = append(block, arith::muli(launch[0], launch[9], location)?)?;
            let index = append(block, arith::addi(offset, launch[3], location)?)?;
            let lhs = append(block, memref::load(memrefs[0], &[index], f32_type, false, None, location)?)?;
            let rhs = append(block, memref::load(memrefs[1], &[index], f32_type, false, None, location)?)?;
            let sum = append(block, arith::addf(lhs, rhs, location)?)?;
            block.append_operation(memref::store(sum, memrefs[2], &[index], false, None, location)?)?;
            Ok(())
        },
    )
}

/// Builds `out = lhs @ rhs` over `f32[MATMUL_M, MATMUL_K] x f32[MATMUL_K, MATMUL_N]` with a two-dimensional grid of
/// `MATMUL_TILE x MATMUL_TILE` thread tiles and an `scf.for` reduction over `MATMUL_K`.
fn tiled_matmul_module<'c, 't>(
    context: &'c Context<'t>,
    compute_capability: (u32, u32),
) -> Result<Module<'c, 't>, Error> {
    let location = context.unknown_location();
    let f32_type = context.float32_type();
    let index_type = context.index_type();
    mosaic_gpu_module(
        context,
        "tiled_matmul",
        compute_capability,
        &[&[MATMUL_M, MATMUL_K], &[MATMUL_K, MATMUL_N], &[MATMUL_M, MATMUL_N]],
        [MATMUL_N / MATMUL_TILE, MATMUL_M / MATMUL_TILE, 1],
        [MATMUL_TILE, MATMUL_TILE, 1],
        |block, memrefs, launch| {
            let row_offset = append(block, arith::muli(launch[1], launch[10], location)?)?;
            let row = append(block, arith::addi(row_offset, launch[4], location)?)?;
            let column_offset = append(block, arith::muli(launch[0], launch[9], location)?)?;
            let column = append(block, arith::addi(column_offset, launch[3], location)?)?;
            let zero = append(block, arith::constant(context.float_attribute(f32_type, 0.0), location)?)?;
            let index_constant = |block: &mut DetachedBlock<'c, 't>, value: usize| {
                append(block, arith::constant(context.integer_attribute(index_type, value as i64), location)?)
            };
            let lower_bound = index_constant(block, 0)?;
            let upper_bound = index_constant(block, MATMUL_K)?;
            let step = index_constant(block, 1)?;

            let mut loop_block = context.block(&[(index_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let k = loop_block.argument(0)?.as_ref();
            let accumulator = loop_block.argument(1)?.as_ref();
            let lhs = append(&mut loop_block, memref::load(memrefs[0], &[row, k], f32_type, false, None, location)?)?;
            let rhs =
                append(&mut loop_block, memref::load(memrefs[1], &[k, column], f32_type, false, None, location)?)?;
            let product = append(&mut loop_block, arith::mulf(lhs, rhs, location)?)?;
            let next = append(&mut loop_block, arith::addf(accumulator, product, location)?)?;
            loop_block.append_operation(scf::r#yield(&[next], location)?)?;
            let result = append(
                block,
                scf::r#for(lower_bound, upper_bound, step, &[zero], false, loop_block.try_into()?, location)?,
            )?;
            block.append_operation(memref::store(result, memrefs[2], &[row, column], false, None, location)?)?;
            Ok(())
        },
    )
}

/// Runs the pinned `mosaic_gpu-serde{serialize=true}` pipeline on `module` and returns the version-zero bytecode that
/// the `mosaic_gpu_v2` runtime consumes, exactly like `core.py::_mosaic_gpu_lowering_rule`.
fn serialize_mosaic_gpu_module(module: &Module<'_, '_>) -> Result<Vec<u8>, Error> {
    let context = module.context();
    context.allow_unregistered_dialects();
    let manager = mosaic_gpu_serde_pass_manager(context, true, None)?;
    if !manager.run(&module.as_operation()?).is_success() {
        return Err(Error::internal("failed to run the Mosaic GPU serialization pass"));
    }
    if !module.verify()? {
        return Err(Error::internal("serialized Mosaic GPU module failed verification"));
    }
    validate_mosaic_gpu_serde_version(module)?;
    module
        .as_operation()?
        .bytecode_for_version(0)
        .ok_or_else(|| Error::internal("failed to write Mosaic GPU bytecode"))
}

/// Parses serialized Mosaic GPU kernel bytecode without text conversion and validates its serde version.
fn parse_mosaic_gpu_kernel<'c, 't>(context: &'c Context<'t>, kernel: &[u8]) -> Result<Module<'c, 't>, Error> {
    context.allow_unregistered_dialects();
    let module = context.parse_module_bytes(kernel)?;
    validate_mosaic_gpu_serde_version(&module)?;
    Ok(module)
}

/// Requires `module` to record the pinned serde version.
fn validate_mosaic_gpu_serde_version(module: &Module<'_, '_>) -> Result<(), Error> {
    match mosaic_gpu_serde_version(module)? {
        Some(version) if version == MOSAIC_GPU_SERDE_VERSION as i64 => Ok(()),
        Some(version) => Err(Error::invalid_argument(format!(
            "Mosaic GPU kernel records serde version {version}, expected {MOSAIC_GPU_SERDE_VERSION}",
        ))),
        None => Err(Error::invalid_argument(
            "Mosaic GPU kernel is missing the `stable_mosaic_gpu.version` module attribute",
        )),
    }
}

/// Returns the SHA-256 kernel hash that `mosaic_gpu_v2` uses to deduplicate compiled kernels.
fn mosaic_gpu_kernel_hash(kernel: &[u8]) -> [u8; 32] {
    Sha256::digest(kernel).into()
}

/// Constructs the `stablehlo.custom_call @mosaic_gpu_v2` operation with the pinned `backend_config` dictionary.
fn mosaic_gpu_custom_call<'c, 't>(
    context: &'c Context<'t>,
    inputs: &[KernelValue<'c, 't>],
    input_shapes: &[&[usize]],
    output_shapes: &[&[usize]],
    kernel: &[u8],
    kernel_hash: &[u8],
    use_custom_barrier: bool,
    location: UnknownLocationRef<'c, 't>,
) -> Result<DetachedCustomCallOperation<'c, 't>, Error> {
    let named = |name: &str, attribute| context.named_attribute(context.identifier(name), attribute);
    let backend_config = context.dictionary_attribute(&[
        named("kernel_hash", context.string_attribute(StringRef::from(kernel_hash)).as_ref()),
        named("module", context.string_attribute(StringRef::from(kernel)).as_ref()),
        named("use_custom_barrier", context.boolean_attribute(use_custom_barrier).as_ref()),
        named("uses_xla_collective_metadata", context.boolean_attribute(false).as_ref()),
    ]);
    let row_major_layout = |shape: &&[usize]| (0..shape.len()).rev().collect::<Vec<_>>();
    let output_types = output_shapes
        .iter()
        .map(|shape| {
            let shape = shape.iter().map(|size| Size::Static(*size)).collect::<Vec<_>>();
            context
                .tensor_type(context.float32_type(), shape.as_slice(), None, location)
                .map(|r#type| r#type.as_ref())
        })
        .collect::<Result<Vec<_>, _>>()?;
    stable_hlo::custom_call(
        inputs,
        MOSAIC_GPU_FFI_TARGET,
        false,
        Some(backend_config.as_ref()),
        CustomCallApiVersion::TypedFfi,
        &[],
        Some(CustomCallMemoryLayouts {
            operands: input_shapes.iter().map(row_major_layout).collect(),
            results: output_shapes.iter().map(row_major_layout).collect(),
        }),
        &[] as &[OutputOperandAliasAttributeRef<'c, 't>],
        None,
        output_types.as_slice(),
        location,
    )
}

/// Constructs the StableHLO program whose `main` forwards its `f32` inputs to one `mosaic_gpu_v2` custom call.
fn mosaic_gpu_program<'c, 't>(
    context: &'c Context<'t>,
    input_shapes: &[&[usize]],
    output_shapes: &[&[usize]],
    kernel: &[u8],
    kernel_hash: &[u8],
    use_custom_barrier: bool,
) -> Result<Module<'c, 't>, Error> {
    let location = context.unknown_location();
    let module = context.module(location)?;
    let tensor_type = |shape: &&[usize]| {
        let shape = shape.iter().map(|size| Size::Static(*size)).collect::<Vec<_>>();
        context
            .tensor_type(context.float32_type(), shape.as_slice(), None, location)
            .map(|r#type| r#type.as_ref())
    };
    let input_types = input_shapes.iter().map(tensor_type).collect::<Result<Vec<TypeRef<'c, 't>>, _>>()?;
    let output_types = output_shapes.iter().map(tensor_type).collect::<Result<Vec<TypeRef<'c, 't>>, _>>()?;
    let mut block = context.block(&input_types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>());
    let inputs = (0..input_shapes.len())
        .map(|index| block.argument(index).map(|argument| argument.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let call = block.append_operation(mosaic_gpu_custom_call(
        context,
        inputs.as_slice(),
        input_shapes,
        output_shapes,
        kernel,
        kernel_hash,
        use_custom_barrier,
        location,
    )?)?;
    let outputs = (0..output_shapes.len())
        .map(|index| call.result(index).map(|result| result.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    block.append_operation(func::r#return(outputs.as_slice(), location)?)?;
    module.body()?.append_operation(func::func(
        "main",
        func::FuncAttributes {
            arguments: input_types.into_iter().map(Into::into).collect(),
            results: output_types.into_iter().map(Into::into).collect(),
            ..Default::default()
        },
        block.try_into()?,
        location,
    )?)?;
    Ok(module)
}

/// Replaces the escaped byte-string values of the named attributes with `<redacted>` so that renderings containing
/// serialized bytecode can be snapshot. MLIR escapes embedded quotes as `\22`, so the first raw quote after the opening
/// one terminates the literal.
fn redact_string_attributes(text: &str, names: &[&str]) -> String {
    let mut text = text.to_string();
    for name in names {
        let key = format!("{name} = \"");
        let start =
            text.find(key.as_str()).unwrap_or_else(|| panic!("missing `{name}` attribute in rendering")) + key.len();
        let end = start + text[start..].find('"').unwrap();
        text.replace_range(start..end, "<redacted>");
    }
    text
}

/// Removes trailing whitespace from every line so renderings can be snapshot exactly; MLIR's `llvm.insertvalue`
/// printer emits a trailing space after the container type.
fn trim_line_ends(text: &str) -> String {
    text.lines().map(str::trim_end).collect::<Vec<_>>().join("\n") + "\n"
}

/// Returns `true` when the NVIDIA-only Mosaic GPU probes are enabled through `RYFT_PJRT_RUN_MOSAIC_GPU_SEAM_PROBE=1`.
fn mosaic_gpu_probe_enabled() -> bool {
    env::var("RYFT_PJRT_RUN_MOSAIC_GPU_SEAM_PROBE").ok().as_deref() == Some("1")
}

/// Returns the compute capability that the CUDA PJRT plugin reports for `device` (e.g., `(9, 0)`).
fn device_compute_capability(device: &Device<'_>) -> (u32, u32) {
    let PjrtValue::String(capability) = device.attribute("compute_capability").unwrap() else {
        panic!("the CUDA PJRT device does not report a string `compute_capability` attribute");
    };
    let (major, minor) = capability.split_once('.').expect("compute capability must be `<major>.<minor>`");
    (major.parse().unwrap(), minor.parse().unwrap())
}

/// One executable Mosaic GPU slice: the kernel module, its host inputs, and the host oracle for the single output.
struct MosaicGpuSlice {
    kernel_module: for<'c, 't> fn(&'c Context<'t>, (u32, u32)) -> Result<Module<'c, 't>, Error>,
    input_shapes: Vec<Vec<usize>>,
    output_shape: Vec<usize>,
    inputs: Vec<Vec<f32>>,
    expected: Vec<f32>,
}

fn vector_add_slice() -> MosaicGpuSlice {
    let lhs = (0..VECTOR_ADD_LENGTH).map(|index| index as f32 * 0.5).collect::<Vec<_>>();
    let rhs = (0..VECTOR_ADD_LENGTH).map(|index| (VECTOR_ADD_LENGTH - index) as f32).collect::<Vec<_>>();
    let expected = lhs.iter().zip(&rhs).map(|(lhs, rhs)| lhs + rhs).collect();
    MosaicGpuSlice {
        kernel_module: vector_add_module,
        input_shapes: vec![vec![VECTOR_ADD_LENGTH], vec![VECTOR_ADD_LENGTH]],
        output_shape: vec![VECTOR_ADD_LENGTH],
        inputs: vec![lhs, rhs],
        expected,
    }
}

fn tiled_matmul_slice() -> MosaicGpuSlice {
    let lhs = (0..MATMUL_M * MATMUL_K)
        .map(|index| ((index / MATMUL_K + index % MATMUL_K) % 7) as f32)
        .collect::<Vec<_>>();
    let rhs = (0..MATMUL_K * MATMUL_N)
        .map(|index| ((index / MATMUL_N * 3 + index % MATMUL_N) % 5) as f32)
        .collect::<Vec<_>>();
    let mut expected = vec![0.0f32; MATMUL_M * MATMUL_N];
    for row in 0..MATMUL_M {
        for column in 0..MATMUL_N {
            for k in 0..MATMUL_K {
                expected[row * MATMUL_N + column] += lhs[row * MATMUL_K + k] * rhs[k * MATMUL_N + column];
            }
        }
    }
    MosaicGpuSlice {
        kernel_module: tiled_matmul_module,
        input_shapes: vec![vec![MATMUL_M, MATMUL_K], vec![MATMUL_K, MATMUL_N]],
        output_shape: vec![MATMUL_M, MATMUL_N],
        inputs: vec![lhs, rhs],
        expected,
    }
}

/// Renders the StableHLO program of `slice` for the compute capability of `device`, returning the program and the
/// serialized kernel bytecode it embeds.
fn slice_program(slice: &MosaicGpuSlice, compute_capability: (u32, u32)) -> (Program, Vec<u8>) {
    let context = Context::new();
    let kernel_module = (slice.kernel_module)(&context, compute_capability).unwrap();
    assert!(kernel_module.verify().unwrap());
    let kernel = serialize_mosaic_gpu_module(&kernel_module).unwrap();
    let input_shapes = slice.input_shapes.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let program = mosaic_gpu_program(
        &context,
        input_shapes.as_slice(),
        &[slice.output_shape.as_slice()],
        kernel.as_slice(),
        &mosaic_gpu_kernel_hash(kernel.as_slice()),
        false,
    )
    .unwrap();
    assert!(program.verify().unwrap());
    (Program::Mlir { bytecode: program.to_string().into_bytes() }, kernel)
}

/// Executes `executable` on `slice`'s inputs and asserts the exact host oracle.
fn execute_slice(
    client: &Client<'_>,
    executable: &crate::LoadedExecutable<'_>,
    device: &Device<'_>,
    slice: &MosaicGpuSlice,
) {
    let buffers = slice
        .inputs
        .iter()
        .zip(&slice.input_shapes)
        .map(|(input, shape)| {
            let bytes = input.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>();
            let dimensions = shape.iter().map(|size| *size as u64).collect::<Vec<_>>();
            client
                .buffer(bytes.as_slice(), BufferType::F32, dimensions.as_slice(), None, device.clone(), None)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let execution_inputs = buffers.into_iter().map(ExecutionInput::from).collect::<Vec<_>>();
    let inputs = ExecutionDeviceInputs::from(execution_inputs.as_slice());
    let mut device_outputs = executable
        .execute(vec![inputs], vec![], 0, None, None, None, None)
        .unwrap()
        .block_until_ready()
        .unwrap()
        .remove(0);
    let output = device_outputs.outputs.remove(0);
    let output_bytes = output.copy_to_host(None).unwrap().r#await().unwrap();
    let output = output_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(output, slice.expected);
}

/// Compiles, executes, serializes, reloads, and re-executes `slice` on the CUDA client, then asserts PTX dump evidence
/// when `MOSAIC_GPU_DUMP_TO` and `MOSAIC_GPU_DUMP_PTX` are set.
fn run_slice_on_cuda(client: &Client<'_>, slice: &MosaicGpuSlice) {
    let device = client.addressable_devices().unwrap().remove(0);
    let compute_capability = device_compute_capability(&device);
    let (program, _) = slice_program(slice, compute_capability);
    let options = test_compilation_options();
    let executable = client.compile(&program, &options).unwrap();
    execute_slice(client, &executable, &device, slice);

    // Ahead-of-time persistence: serialize the enclosing executable, reload it, and execute the reloaded copy.
    let serialized = executable.executable().unwrap().serialize().unwrap();
    let reloaded = client
        .deserialize_and_load_executable(serialized.data(), Some(&options), &LoadOptions::default())
        .unwrap();
    execute_slice(client, &reloaded, &device, slice);

    if let (Ok(dump_directory), Ok(_)) = (env::var("MOSAIC_GPU_DUMP_TO"), env::var("MOSAIC_GPU_DUMP_PTX")) {
        let target = format!(".target sm_{}{}", compute_capability.0, compute_capability.1);
        let ptx_files = ptx_dumps(Path::new(dump_directory.as_str()));
        assert!(!ptx_files.is_empty(), "no `.ptx` dump was written to `{dump_directory}`");
        assert!(
            ptx_files.iter().any(|ptx| ptx.contains(target.as_str())),
            "no dumped PTX names the device target `{target}` in `{dump_directory}`",
        );
    }
}

/// Returns the contents of every `.ptx` file below `directory`.
fn ptx_dumps(directory: &Path) -> Vec<String> {
    let mut contents = Vec::new();
    for entry in
        fs::read_dir(directory).unwrap_or_else(|error| panic!("failed to read `{}`: {error}", directory.display()))
    {
        let path = entry.unwrap().path();
        if path.is_dir() {
            contents.extend(ptx_dumps(&path));
        } else if path.extension().is_some_and(|extension| extension == "ptx") {
            contents.push(fs::read_to_string(&path).unwrap());
        }
    }
    contents
}

/// Joins rendered lines, each given as one or more literal fragments, into one exact multi-line rendering so long
/// canonical MLIR lines stay within the source column limit.
macro_rules! rendering {
    ($( [$($fragment:literal),+ $(,)?] ),+ $(,)?) => {
        concat!($( $($fragment,)+ "\n",)+)
    };
}

#[test]
fn test_vector_add_module() {
    use pretty_assertions::assert_eq;

    let context = Context::new();
    let module = vector_add_module(&context, PORTABLE_COMPUTE_CAPABILITY).unwrap();
    assert!(module.verify().unwrap());
    assert_eq!(
        trim_line_ends(module.to_string().as_str()),
        rendering!(
            ["module @vector_add attributes {mosaic_gpu.arch_major = 9 : i32, mosaic_gpu.arch_minor = 0 : i32} {"],
            [
                "  func.func private @mosaic_gpu_init_tma_desc(!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, ",
                "!llvm.ptr, i64, !llvm.ptr)",
            ],
            ["  llvm.mlir.global external @global_scratch() {addr_space = 4 : i32} : !llvm.array<0 x i8>"],
            [
                "  func.func @vector_add_mosaic_gpu(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes ",
                "{llvm.emit_c_interface} {",
            ],
            ["    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !gpu.async.token"],
            ["    %1 = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr"],
            ["    %2 = llvm.load %1 : !llvm.ptr -> !llvm.ptr"],
            ["    %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"],
            ["    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"],
            ["    %5 = llvm.insertvalue %2, %4[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"],
            ["    %6 = llvm.mlir.constant(0 : i64) : i64"],
            ["    %7 = llvm.insertvalue %6, %5[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"],
            ["    %8 = llvm.mlir.constant(1024 : i64) : i64"],
            ["    %9 = llvm.insertvalue %8, %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"],
            ["    %10 = llvm.mlir.constant(1 : i64) : i64"],
            [
                "    %11 = llvm.insertvalue %10, %9[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 ",
                "x i64>)>",
            ],
            [
                "    %12 = builtin.unrealized_conversion_cast %11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, ",
                "array<1 x i64>)> to memref<1024xf32> {mosaic_gpu.from_kernel_arg_idx = 0 : i32, ",
                "mosaic_gpu.original_kernel_arg}",
            ],
            ["    %13 = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr"],
            ["    %14 = llvm.load %13 : !llvm.ptr -> !llvm.ptr"],
            ["    %15 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"],
            ["    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"],
            ["    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"],
            ["    %18 = llvm.mlir.constant(0 : i64) : i64"],
            ["    %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"],
            ["    %20 = llvm.mlir.constant(1024 : i64) : i64"],
            [
                "    %21 = llvm.insertvalue %20, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, ",
                "array<1 x i64>)>",
            ],
            ["    %22 = llvm.mlir.constant(1 : i64) : i64"],
            [
                "    %23 = llvm.insertvalue %22, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, ",
                "array<1 x i64>)>",
            ],
            [
                "    %24 = builtin.unrealized_conversion_cast %23 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, ",
                "array<1 x i64>)> to memref<1024xf32> {mosaic_gpu.from_kernel_arg_idx = 1 : i32, ",
                "mosaic_gpu.original_kernel_arg}",
            ],
            ["    %25 = llvm.getelementptr %arg1[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr"],
            ["    %26 = llvm.load %25 : !llvm.ptr -> !llvm.ptr"],
            ["    %27 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"],
            ["    %28 = llvm.insertvalue %26, %27[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"],
            ["    %29 = llvm.insertvalue %26, %28[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"],
            ["    %30 = llvm.mlir.constant(0 : i64) : i64"],
            ["    %31 = llvm.insertvalue %30, %29[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"],
            ["    %32 = llvm.mlir.constant(1024 : i64) : i64"],
            [
                "    %33 = llvm.insertvalue %32, %31[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, ",
                "array<1 x i64>)>",
            ],
            ["    %34 = llvm.mlir.constant(1 : i64) : i64"],
            [
                "    %35 = llvm.insertvalue %34, %33[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, ",
                "array<1 x i64>)>",
            ],
            [
                "    %36 = builtin.unrealized_conversion_cast %35 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, ",
                "array<1 x i64>)> to memref<1024xf32> {mosaic_gpu.from_kernel_arg_idx = 2 : i32, ",
                "mosaic_gpu.original_kernel_arg}",
            ],
            ["    %c4 = arith.constant 4 : index"],
            ["    %c1 = arith.constant 1 : index"],
            ["    %c1_0 = arith.constant 1 : index"],
            ["    %c256 = arith.constant 256 : index"],
            ["    %c1_1 = arith.constant 1 : index"],
            ["    %c1_2 = arith.constant 1 : index"],
            ["    %c0_i32 = arith.constant 0 : i32"],
            [
                "    %37 = gpu.launch async [%0] blocks(%arg2, %arg3, %arg4) in (%arg8 = %c4, %arg9 = %c1, ",
                "%arg10 = %c1_0) threads(%arg5, %arg6, %arg7) in (%arg11 = %c256, %arg12 = %c1_1, %arg13 = ",
                "%c1_2) dynamic_shared_memory_size %c0_i32 {",
            ],
            ["      %38 = arith.muli %arg2, %arg11 : index"],
            ["      %39 = arith.addi %38, %arg5 : index"],
            ["      %40 = memref.load %12[%39] : memref<1024xf32>"],
            ["      %41 = memref.load %24[%39] : memref<1024xf32>"],
            ["      %42 = arith.addf %40, %41 : f32"],
            ["      memref.store %42, %36[%39] : memref<1024xf32>"],
            ["      gpu.terminator"],
            ["    }"],
            ["    return"],
            ["  }"],
            ["}"],
        ),
    );
}

#[test]
fn test_tiled_matmul_module() {
    use pretty_assertions::assert_eq;

    let context = Context::new();
    let module = tiled_matmul_module(&context, PORTABLE_COMPUTE_CAPABILITY).unwrap();
    assert!(module.verify().unwrap());
    let rendering = trim_line_ends(module.to_string().as_str());
    // The prologue follows the vector-add module with rank-two descriptors; snapshot the launch body that differs.
    let body_start = rendering.find("%49 = gpu.launch").unwrap();
    assert_eq!(
        &rendering[body_start..],
        rendering!(
            [
                "%49 = gpu.launch async [%0] blocks(%arg2, %arg3, %arg4) in (%arg8 = %c3, %arg9 = %c2, %arg10 = ",
                "%c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c16, %arg12 = %c16_0, %arg13 = %c1_1) ",
                "dynamic_shared_memory_size %c0_i32 {",
            ],
            ["      %50 = arith.muli %arg3, %arg12 : index"],
            ["      %51 = arith.addi %50, %arg6 : index"],
            ["      %52 = arith.muli %arg2, %arg11 : index"],
            ["      %53 = arith.addi %52, %arg5 : index"],
            ["      %cst = arith.constant 0.000000e+00 : f32"],
            ["      %c0 = arith.constant 0 : index"],
            ["      %c8 = arith.constant 8 : index"],
            ["      %c1_2 = arith.constant 1 : index"],
            ["      %54 = scf.for %arg14 = %c0 to %c8 step %c1_2 iter_args(%arg15 = %cst) -> (f32) {"],
            ["        %55 = memref.load %16[%51, %arg14] : memref<32x8xf32>"],
            ["        %56 = memref.load %32[%arg14, %53] : memref<8x48xf32>"],
            ["        %57 = arith.mulf %55, %56 : f32"],
            ["        %58 = arith.addf %arg15, %57 : f32"],
            ["        scf.yield %58 : f32"],
            ["      }"],
            ["      memref.store %54, %48[%51, %53] : memref<32x48xf32>"],
            ["      gpu.terminator"],
            ["    }"],
            ["    return"],
            ["  }"],
            ["}"],
        ),
    );
    assert!(rendering.starts_with(
        "module @tiled_matmul attributes {mosaic_gpu.arch_major = 9 : i32, mosaic_gpu.arch_minor = 0 : i32} {\n",
    ));
    assert!(rendering.contains("to memref<32x8xf32> {mosaic_gpu.from_kernel_arg_idx = 0 : i32"));
    assert!(rendering.contains("to memref<8x48xf32> {mosaic_gpu.from_kernel_arg_idx = 1 : i32"));
    assert!(rendering.contains("to memref<32x48xf32> {mosaic_gpu.from_kernel_arg_idx = 2 : i32"));
}

#[test]
fn test_serialize_mosaic_gpu_module() {
    use pretty_assertions::assert_eq;

    let context = Context::new();
    let module = vector_add_module(&context, PORTABLE_COMPUTE_CAPABILITY).unwrap();
    assert_eq!(mosaic_gpu_serde_version(&module), Ok(None));
    let kernel = serialize_mosaic_gpu_module(&module).unwrap();
    assert_eq!(mosaic_gpu_serde_version(&module), Ok(Some(MOSAIC_GPU_SERDE_VERSION as i64)));
    assert!(kernel.starts_with(b"ML\xefR"), "Mosaic GPU kernels are MLIR bytecode");
    assert_eq!(mosaic_gpu_kernel_hash(kernel.as_slice()).len(), 32);

    // Binary round trip without text conversion re-emits byte-identical bytecode and preserves the version.
    let parsed = parse_mosaic_gpu_kernel(&context, kernel.as_slice()).unwrap();
    assert!(parsed.verify().unwrap());
    assert_eq!(parsed.as_operation().unwrap().bytecode_for_version(0).unwrap(), kernel);
    assert_eq!(parsed.to_string(), module.to_string());

    // Garbage bytes and modules that were never serialized are rejected.
    assert!(matches!(
        parse_mosaic_gpu_kernel(&context, b"not mlir bytecode"),
        Err(Error::ParsingError { message, .. }) if message == "failed to parse MLIR module",
    ));
    let unserialized = vector_add_module(&context, PORTABLE_COMPUTE_CAPABILITY).unwrap();
    let unserialized = unserialized.as_operation().unwrap().bytecode_for_version(0).unwrap();
    assert!(matches!(
        parse_mosaic_gpu_kernel(&context, unserialized.as_slice()),
        Err(Error::InvalidArgument { message, .. })
            if message == "Mosaic GPU kernel is missing the `stable_mosaic_gpu.version` module attribute",
    ));
}

#[test]
fn test_mosaic_gpu_custom_call() {
    use pretty_assertions::assert_eq;

    let context = Context::new();
    let location = context.unknown_location();
    let kernel = b"kernel bytes with \"quotes\" and \0 nul".to_vec();
    let kernel_hash = mosaic_gpu_kernel_hash(kernel.as_slice());
    let input_type = context
        .tensor_type(context.float32_type(), &[Size::Static(4), Size::Static(2)], None, location)
        .unwrap();
    let block = context.block(&[(input_type, location)]);
    let input = block.argument(0).unwrap().as_ref();
    let call = mosaic_gpu_custom_call(
        &context,
        &[input],
        &[&[4, 2]],
        &[&[4]],
        kernel.as_slice(),
        &kernel_hash,
        false,
        location,
    )
    .unwrap();
    assert_eq!(call.custom_call_target_name().unwrap().as_str().unwrap(), "mosaic_gpu_v2");
    assert_eq!(call.custom_call_api_version().unwrap(), CustomCallApiVersion::TypedFfi);
    assert_eq!(call.custom_call_has_side_effect().unwrap(), false);
    let layouts = call.custom_call_memory_layouts().unwrap().unwrap();
    assert_eq!(layouts.operands, vec![vec![1, 0]]);
    assert_eq!(layouts.results, vec![vec![0]]);

    // The backend configuration carries the exact pinned dictionary, including the raw bytecode bytes.
    let backend_config = call.custom_call_backend_config().unwrap();
    let backend_config = backend_config.cast::<ryft_mlir::DictionaryAttributeRef>().unwrap();
    assert_eq!(
        backend_config.elements().map(|element| element.name().to_string()).collect::<Vec<_>>(),
        ["kernel_hash", "module", "use_custom_barrier", "uses_xla_collective_metadata"],
    );
    let string_bytes = |name: &str| {
        let attribute = backend_config.element_by_name(name).unwrap().unwrap();
        attribute.cast::<ryft_mlir::StringAttributeRef>().unwrap().string().bytes().to_vec()
    };
    let boolean = |name: &str| {
        let attribute = backend_config.element_by_name(name).unwrap().unwrap();
        attribute.cast::<ryft_mlir::BooleanAttributeRef>().unwrap().value()
    };
    assert_eq!(string_bytes("kernel_hash"), kernel_hash.to_vec());
    assert_eq!(string_bytes("module"), kernel);
    assert!(!boolean("use_custom_barrier"));
    assert!(!boolean("uses_xla_collective_metadata"));
}

#[test]
fn test_mosaic_gpu_program() {
    use pretty_assertions::assert_eq;

    let context = Context::new();
    let module = vector_add_module(&context, PORTABLE_COMPUTE_CAPABILITY).unwrap();
    let kernel = serialize_mosaic_gpu_module(&module).unwrap();
    let kernel_hash = mosaic_gpu_kernel_hash(kernel.as_slice());
    let shape: &[usize] = &[VECTOR_ADD_LENGTH];
    let program =
        mosaic_gpu_program(&context, &[shape, shape], &[shape], kernel.as_slice(), &kernel_hash, false).unwrap();
    assert!(program.verify().unwrap());
    assert_eq!(
        redact_string_attributes(program.to_string().as_str(), &["kernel_hash", "module"]),
        rendering!(
            ["module {"],
            ["  func.func @main(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {"],
            [
                "    %0 = stablehlo.custom_call @mosaic_gpu_v2(%arg0, %arg1) {api_version = 4 : i32, ",
                "backend_config = {kernel_hash = \"<redacted>\", module = \"<redacted>\", use_custom_barrier = ",
                "false, uses_xla_collective_metadata = false}, operand_layouts = [dense<0> : tensor<1xindex>, ",
                "dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : ",
                "(tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>",
            ],
            ["    return %0 : tensor<1024xf32>"],
            ["  }"],
            ["}"],
        ),
    );

    // The rendering embeds the raw hash and bytecode as escaped MLIR strings, so parsing it back recovers both.
    let parsed = context.parse_module(program.to_string()).unwrap();
    assert!(parsed.verify().unwrap());
    assert_eq!(parsed.to_string(), program.to_string());
}

#[test]
fn test_mosaic_gpu_vector_add_on_cuda_pjrt() {
    if !mosaic_gpu_probe_enabled() {
        return;
    }
    let mut tested_cuda = false;
    test_for_each_platform!(|_plugin, client, platform| {
        if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
            tested_cuda = true;
            run_slice_on_cuda(&client, &vector_add_slice());
        }
    });
    assert!(tested_cuda, "Mosaic GPU seam probe requires `ryft-experimental` to be built with a `cuda-*` feature");
}

#[test]
fn test_mosaic_gpu_tiled_matmul_on_cuda_pjrt() {
    if !mosaic_gpu_probe_enabled() {
        return;
    }
    let mut tested_cuda = false;
    test_for_each_platform!(|_plugin, client, platform| {
        if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
            tested_cuda = true;
            run_slice_on_cuda(&client, &tiled_matmul_slice());
        }
    });
    assert!(tested_cuda, "Mosaic GPU seam probe requires `ryft-experimental` to be built with a `cuda-*` feature");
}

#[test]
fn test_mosaic_gpu_diagnostics_on_cuda_pjrt() {
    if !mosaic_gpu_probe_enabled() {
        return;
    }
    let mut tested_cuda = false;
    test_for_each_platform!(|_plugin, client, platform| {
        if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
            tested_cuda = true;
            let device = client.addressable_devices().unwrap().remove(0);
            let slice = vector_add_slice();
            let (_, kernel) = slice_program(&slice, device_compute_capability(&device));
            let kernel_hash = mosaic_gpu_kernel_hash(kernel.as_slice());
            let shape: &[usize] = &[VECTOR_ADD_LENGTH];
            let compile_error = |kernel: &[u8], kernel_hash: &[u8], use_custom_barrier: bool| {
                let context = Context::new();
                let program =
                    mosaic_gpu_program(&context, &[shape, shape], &[shape], kernel, kernel_hash, use_custom_barrier)
                        .unwrap();
                let program = Program::Mlir { bytecode: program.to_string().into_bytes() };
                client
                    .compile(&program, &test_compilation_options())
                    .err()
                    .expect("compilation must fail")
                    .to_string()
            };

            // `InstantiateResources` in `jaxlib/mosaic/gpu/custom_call.cc` checks the barrier flag, then the hash
            // length, and only then parses the serialized module.
            assert!(
                compile_error(kernel.as_slice(), &kernel_hash, true)
                    .contains("Custom barrier is not supported on GPUs."),
            );
            assert!(
                compile_error(kernel.as_slice(), &kernel_hash[..31], false)
                    .contains("Kernel hash size is 31 bytes, expected 32 bytes"),
            );
            assert!(
                compile_error(b"not mlir bytecode", &kernel_hash, false).contains("Failed to parse Mosaic GPU module"),
            );
        }
    });
    assert!(tested_cuda, "Mosaic GPU seam probe requires `ryft-experimental` to be built with a `cuda-*` feature");
}

#[test]
fn test_mosaic_gpu_repeated_compilation_on_cuda_pjrt() {
    if !mosaic_gpu_probe_enabled() {
        return;
    }
    let mut tested_cuda = false;
    test_for_each_platform!(|_plugin, client, platform| {
        if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
            tested_cuda = true;
            // Repeated compile/execute cycles give the CI sanitizer step repeated kernel lifetimes to observe.
            let device = client.addressable_devices().unwrap().remove(0);
            let slice = vector_add_slice();
            for _ in 0..8 {
                let (program, _) = slice_program(&slice, device_compute_capability(&device));
                let executable = client.compile(&program, &test_compilation_options()).unwrap();
                execute_slice(&client, &executable, &device, &slice);
            }
        }
    });
    assert!(tested_cuda, "Mosaic GPU seam probe requires `ryft-experimental` to be built with a `cuda-*` feature");
}
