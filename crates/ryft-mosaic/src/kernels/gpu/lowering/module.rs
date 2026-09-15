//! Typed host-buffer ABI and versioned source serialization for Mosaic GPU kernels.

use ryft_core::{ArrayAddressing, ArrayType, DataType};
use ryft_mlir::dialects::gpu::{Dim3, LaunchProperties};
use ryft_mlir::dialects::llvm::Linkage;
use ryft_mlir::dialects::mosaic::gpu::{mosaic_gpu_serde_pass_manager, mosaic_gpu_serde_version};
use ryft_mlir::dialects::{arith, builtin, func, gpu, llvm};
use ryft_mlir::{
    Attribute, Block, Context, DetachedBlock, DialectHandle, Module, Operation, Size, SymbolVisibility, Type, TypeRef,
    UnknownLocationRef, Value, WalkOrder, WalkResult,
};
use ryft_xla_sys::mlir::dialects::mosaic::gpu::MOSAIC_GPU_SERDE_VERSION;

use crate::kernels::gpu::lowering::validation::checked_array;
use crate::kernels::gpu::lowering::{KernelValue, append};
use crate::kernels::gpu::{Error, Target};

/// Attribute identifying a native buffer argument in the pinned Mosaic lowering.
const KERNEL_ARGUMENT_INDEX_ATTRIBUTE: &str = "mosaic_gpu.from_kernel_arg_idx";
/// Attribute identifying the original host memref cast before native lowering.
const ORIGINAL_KERNEL_ARGUMENT_ATTRIBUTE: &str = "mosaic_gpu.original_kernel_arg";

/// Loads kernel argument `index` from the `buffers` pointer array and wraps it in a row-major typed memref
/// descriptor exactly like `utils.getelementptr` followed by `utils.ptr_as_memref` in the pinned JAX lowering.
fn kernel_argument<'c, 't>(
    context: &'c Context<'t>,
    block: &mut DetachedBlock<'c, 't>,
    buffers: KernelValue<'c, 't>,
    index: usize,
    r#type: &ArrayType,
    location: UnknownLocationRef<'c, 't>,
) -> Result<KernelValue<'c, 't>, Error> {
    let shape = checked_shape(r#type)?;
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

    let memref_type = memref_type(context, r#type, false)?;
    let mut cast = builtin::unrealized_conversion_cast(&[descriptor], &[memref_type.as_ref()], location)?;
    cast.set_attribute(KERNEL_ARGUMENT_INDEX_ATTRIBUTE, context.integer_attribute(i32_type, index as i64));
    cast.set_attribute(ORIGINAL_KERNEL_ARGUMENT_ATTRIBUTE, context.unit_attribute());
    append(block, cast)
}

/// Builds a Mosaic GPU module following the pinned JAX host ABI.
///
/// `body` receives the launch block, global argument memrefs, shared attribution memrefs, and TMA descriptor pointers
/// in declaration order. Each `tma_sources` entry indexes `arguments`; host initialization captures its checked whole
/// window descriptor by value through the pinned native Mosaic ABI.
/// The first twelve block arguments are `block_id.{x,y,z}`, `thread_id.{x,y,z}`, `grid_dim.{x,y,z}`, and
/// `block_dim.{x,y,z}`, respectively. Clustered launches append cluster IDs and the grid dimensions in clusters
/// before shared attributions. In particular, arguments 15–17 lower to PTX `%nclusterid`, not the number of CTAs
/// per cluster. Physical grid dimensions count CTAs and must be divisible by the target's cluster dimensions.
/// This function terminates the body.
/// Each global argument pairs its native pointer-array slot with its static dense row-major type. Untiled explicit
/// row-major layouts are accepted. Global and shared memrefs are flattened to one dimension, including scalar arrays
/// as a one-element memref. `shared_alignments` supplies one validated alignment per attribution, or an empty slice
/// for the 16-byte baseline. Alignment is transferred to the actual outlined shared-memory globals.
pub(super) fn build<'c, 't, B>(
    context: &'c Context<'t>,
    target: &Target,
    kernel_name: &str,
    arguments: &[(usize, ArrayType)],
    shared_storage_types: &[ArrayType],
    shared_alignments: &[usize],
    tma_sources: &[usize],
    grid: [usize; 3],
    body: B,
) -> Result<Module<'c, 't>, Error>
where
    B: FnOnce(
        &mut DetachedBlock<'c, 't>,
        &[KernelValue<'c, 't>],
        &[KernelValue<'c, 't>],
        &[KernelValue<'c, 't>],
    ) -> Result<(), Error>,
{
    if arguments.len() > i32::MAX as usize
        || arguments.iter().any(|(slot, _)| *slot > i32::MAX as usize)
        || grid.iter().any(|size| *size == 0 || *size > i64::MAX as usize)
    {
        return Err(Error::Invalid {
            message: "kernel argument count or launch grid exceeds the native ABI".to_owned(),
        });
    }
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    let compute_capability = target.compute_capability();
    let clustered = target.blocks_per_cluster() != 1;
    if grid[0] % target.blocks_per_cluster() as usize != 0 {
        return Err(Error::Invalid { message: "physical grid must be divisible by the cluster dimensions".to_owned() });
    }
    let block = [target.threads_per_block() as usize, 1, 1];
    if !shared_alignments.is_empty() && shared_alignments.len() != shared_storage_types.len()
        || shared_alignments.iter().any(|alignment| !matches!(alignment, 16 | 128 | 256))
    {
        return Err(Error::Invalid {
            message: "shared attribution alignments must match storage and be 16, 128, or 256 bytes".to_owned(),
        });
    }
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
    let mut memrefs = Vec::with_capacity(arguments.len());
    for (index, r#type) in arguments {
        memrefs.push(kernel_argument(context, &mut function_block, buffers, *index, r#type, location)?);
    }
    let descriptors = tma_descriptors(context, &mut function_block, buffers, arguments, tma_sources, location)?;
    let mut dimensions = Vec::with_capacity(6);
    for size in grid.into_iter().chain(block) {
        let size = context.integer_attribute(index_type, size as i64);
        dimensions.push(append(&mut function_block, arith::constant(size, location)?)?);
    }
    let dynamic_shared_memory_size =
        append(&mut function_block, arith::constant(context.integer_attribute(i32_type, 0), location)?)?;

    let cluster_dimensions = if clustered {
        let mut dimensions = Vec::with_capacity(3);
        for size in [target.blocks_per_cluster(), 1, 1] {
            dimensions.push(append(
                &mut function_block,
                arith::constant(context.integer_attribute(index_type, size as i64), location)?,
            )?);
        }
        Some(Dim3 { x: dimensions[0], y: dimensions[1], z: dimensions[2] })
    } else {
        None
    };
    let configuration_count = if clustered { 18 } else { 12 };
    let mut launch_types = vec![(index_type.as_ref(), location); configuration_count];
    for r#type in shared_storage_types {
        launch_types.push((memref_type(context, r#type, true)?, location));
    }
    let mut launch_block = context.block(&launch_types);
    let shared = (configuration_count..launch_types.len())
        .map(|index| launch_block.argument(index).map(|argument| argument.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let descriptors = descriptors
        .into_iter()
        .map(|descriptor| {
            append(&mut launch_block, builtin::unrealized_conversion_cast(&[descriptor], &[pointer_type], location)?)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    body(&mut launch_block, &memrefs, &shared, &descriptors)?;
    launch_block.append_operation(gpu::terminator(location)?)?;
    let mut launch = function_block.append_operation(gpu::launch(
        LaunchProperties {
            async_dependencies: vec![token],
            grid_size: Dim3 { x: dimensions[0], y: dimensions[1], z: dimensions[2] },
            block_size: Dim3 { x: dimensions[3], y: dimensions[4], z: dimensions[5] },
            cluster_size: cluster_dimensions,
            dynamic_shared_memory_size: Some(dynamic_shared_memory_size),
            module: None,
            function: None,
            workgroup_attributions: Some(shared_storage_types.len()),
            async_object: None,
            cooperative: false,
            is_async: true,
        },
        launch_block.try_into()?,
        location,
    )?)?;
    let alignments = (0..shared_storage_types.len())
        .map(|index| {
            context.dictionary_attribute(&[context.named_attribute(
                context.identifier("llvm.align"),
                context.integer_attribute(
                    context.signless_integer_type(64),
                    shared_alignments.get(index).copied().unwrap_or(16) as i64,
                ),
            )])
        })
        .collect::<Vec<_>>();
    launch.set_attribute(gpu::WORKGROUP_ATTRIBUTION_ATTRIBUTES_ATTRIBUTE, context.array_attribute(&alignments));
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

/// Initializes CUDA tensor maps on the host and captures each aligned descriptor by value in the GPU launch.
/// The pinned Mosaic outlining pass converts these LLVM array captures to native by-value pointer arguments.
fn tma_descriptors<'c, 't>(
    context: &'c Context<'t>,
    block: &mut DetachedBlock<'c, 't>,
    buffers: KernelValue<'c, 't>,
    arguments: &[(usize, ArrayType)],
    sources: &[usize],
    location: UnknownLocationRef<'c, 't>,
) -> Result<Vec<KernelValue<'c, 't>>, Error> {
    let pointer_type = context.llvm_pointer_type(0)?.as_ref();
    let integer_type = context.signless_integer_type(64);
    let descriptor_type = context.llvm_array_type(context.signless_integer_type(8), 128)?.as_ref();
    let constant = |block: &mut DetachedBlock<'c, 't>, value: usize| {
        append(block, llvm::constant(context.integer_attribute(integer_type, value as i64), integer_type, location)?)
    };
    let address = |block: &mut DetachedBlock<'c, 't>, base, element_type: TypeRef<'c, 't>, index: usize| {
        append(
            block,
            llvm::get_element_ptr(
                base,
                &[],
                pointer_type,
                context.dense_i32_array_attribute(&[index as i32])?.as_ref(),
                context.type_attribute(element_type).as_ref(),
                None,
                location,
            )?,
        )
    };
    let mut descriptors = Vec::with_capacity(sources.len());
    for &source in sources {
        let (slot, r#type) = arguments.get(source).ok_or_else(|| Error::Invalid {
            message: "TMA source parameter exceeds the native argument list".to_owned(),
        })?;
        super::memory::validate_tma_type(r#type)?;
        let shape = r#type.static_shape().unwrap();
        let shape = shape.dimensions();
        let one = constant(block, 1)?;
        let descriptor = append(block, llvm::alloca(one, descriptor_type, pointer_type, Some(64), false, location)?)?;
        let source_slot = address(block, buffers, pointer_type, *slot)?;
        let source_pointer = append(block, llvm::load(source_slot, pointer_type, None, false, location)?)?;
        let count = constant(block, shape.len())?;
        let sizes = append(block, llvm::alloca(count, integer_type, pointer_type, Some(8), false, location)?)?;
        let strides = append(block, llvm::alloca(count, integer_type, pointer_type, Some(8), false, location)?)?;
        for (axis, &extent) in shape.iter().enumerate() {
            let size = constant(block, extent)?;
            let size_pointer = address(block, sizes, integer_type.as_ref(), axis)?;
            block.append_operation(llvm::store(size, size_pointer, None, false, location)?)?;
            let stride = constant(block, shape[axis + 1..].iter().product())?;
            let stride_pointer = address(block, strides, integer_type.as_ref(), axis)?;
            block.append_operation(llvm::store(stride, stride_pointer, None, false, location)?)?;
        }
        let element = constant(
            block,
            match r#type.data_type() {
                DataType::U32 => 3,
                DataType::U64 | DataType::F64 => 4,
                DataType::F16 => 5,
                DataType::F32 => 6,
                DataType::BF16 => 7,
                DataType::I32 => 9,
                DataType::I64 => 10,
                _ => unreachable!(),
            },
        )?;
        let swizzle = constant(block, 16)?;
        block.append_operation(func::call(
            "mosaic_gpu_init_tma_desc",
            func::CallProperties {
                arguments: [descriptor, source_pointer, element, count, sizes, strides, swizzle, sizes]
                    .into_iter()
                    .map(Into::into)
                    .collect(),
                ..Default::default()
            },
            location,
        )?)?;
        descriptors.push(append(block, llvm::load(descriptor, descriptor_type, Some(64), false, location)?)?);
    }
    Ok(descriptors)
}

/// Converts an admitted scalar data type to its native storage element type. Integer signedness remains an operation
/// property; all integer memrefs use signless MLIR integers. Boolean storage uses `i1`, whose memref allocation uses
/// one byte per element rather than packed bits.
pub(super) fn element_type<'c, 't>(context: &'c Context<'t>, data_type: DataType) -> Result<TypeRef<'c, 't>, Error> {
    Ok(match data_type {
        DataType::Boolean => context.signless_integer_type(1).as_ref(),
        DataType::U8 | DataType::F8E4M3FN | DataType::F8E8M0FNU => context.signless_integer_type(8).as_ref(),
        DataType::I32 | DataType::U32 => context.signless_integer_type(32).as_ref(),
        DataType::I64 | DataType::U64 => context.signless_integer_type(64).as_ref(),
        DataType::F16 => context.float16_type().as_ref(),
        DataType::BF16 => context.bfloat16_type().as_ref(),
        DataType::F32 => context.float32_type().as_ref(),
        DataType::F64 => context.float64_type().as_ref(),
        _ => {
            return Err(Error::Unsupported {
                operation: "buffer",
                reason: format!("unsupported native storage data type `{data_type}`"),
            });
        }
    })
}

/// Creates a checked dense memref with either default global addressing or CUDA shared-memory address space `3`.
pub(super) fn memref_type<'c, 't>(
    context: &'c Context<'t>,
    r#type: &ArrayType,
    shared: bool,
) -> Result<TypeRef<'c, 't>, Error> {
    let shape = checked_shape(r#type)?.into_iter().map(Size::Static).collect::<Vec<_>>();
    let memory_space = shared.then(|| context.integer_attribute(context.signless_integer_type(64), 3).as_ref());
    Ok(context
        .mem_ref_type(
            element_type(context, r#type.data_type())?,
            &shape,
            None,
            memory_space,
            context.unknown_location(),
        )?
        .as_ref())
}

/// Serializes verified typed IR with the exact pinned Mosaic source version. Serialization mutates the module into
/// stable dialect operations; callers retain binary bytes, not a second mutable source representation.
pub(in crate::kernels::gpu) fn serialize(module: &Module<'_, '_>) -> Result<Vec<u8>, Error> {
    if !module.verify()? {
        return Err(ryft_mlir::Error::internal("mosaic GPU source module failed verification").into());
    }
    let context = module.context();
    // The native pipeline repeats outlining, which is a no-op once launches have been replaced. Explicit function
    // attribution alignment survives GPU-to-NVVM conversion as alignment on the actual shared-memory globals.
    // GPU outlining does not forward attribution dictionaries. This builder owns exactly one launch; carry its
    // canonical alignment attributes across that pass rather than assuming every shared allocation has one alignment.
    let mut launch_alignments = Vec::new();
    module.as_operation()?.walk(WalkOrder::PreOrder, |operation| {
        if operation.name().as_str() == Ok("gpu.launch") {
            launch_alignments.push(operation.attribute(gpu::WORKGROUP_ATTRIBUTION_ATTRIBUTES_ATTRIBUTE));
        }
        WalkResult::Advance
    });
    let launch_alignments = launch_alignments.into_iter().collect::<Result<Vec<_>, _>>()?;
    if launch_alignments.len() > 1 {
        return Err(ryft_mlir::Error::internal("Mosaic GPU source must contain at most one launch").into());
    }
    let mut outlining = context.pass_manager()?;
    outlining.add_pass(gpu::create_gpu_kernel_outlining_pass()?);
    if !outlining.run(&module.as_operation()?).is_success() {
        return Err(ryft_mlir::Error::internal("failed to outline Mosaic GPU kernels").into());
    }
    let mut functions = Vec::new();
    module.as_operation()?.walk(WalkOrder::PreOrder, |operation| {
        if operation.name().as_str() == Ok("gpu.func") {
            functions.push(operation);
        }
        WalkResult::Advance
    });
    for mut function in functions {
        if let Some(Some(alignments)) = launch_alignments.first() {
            function.set_attribute(gpu::WORKGROUP_ATTRIBUTION_ATTRIBUTES_ATTRIBUTE, *alignments);
        }
    }
    // The pinned runtime loads NVGPU only after parsing bytecode, so its token types would remain opaque during
    // deserialization. Lower them to the native NVVM copy/group/wait instructions before versioned transport.
    let mut asynchronous = context.pass_manager()?;
    asynchronous.add_pass(builtin::create_conversion_nvgpu_to_nvvm_pass()?);
    if !asynchronous.run(&module.as_operation()?).is_success() {
        return Err(
            ryft_mlir::Error::internal("failed to lower Mosaic GPU asynchronous tokens before serialization").into()
        );
    }
    context.allow_unregistered_dialects();
    let manager = mosaic_gpu_serde_pass_manager(context, true, None)?;
    if !manager.run(&module.as_operation()?).is_success() {
        return Err(ryft_mlir::Error::internal("failed to run the Mosaic GPU serialization pass").into());
    }
    if !module.verify()? {
        return Err(ryft_mlir::Error::internal("serialized Mosaic GPU module failed verification").into());
    }
    if mosaic_gpu_serde_version(module)? != Some(MOSAIC_GPU_SERDE_VERSION as i64) {
        return Err(ryft_mlir::Error::internal("serialized Mosaic GPU module has an unexpected serde version").into());
    }
    module
        .as_operation()?
        .bytecode_for_version(0)
        .ok_or_else(|| ryft_mlir::Error::internal("failed to write Mosaic GPU bytecode").into())
}

/// Validates static dense storage and returns its flat element count within the native signed integer range.
fn checked_shape(r#type: &ArrayType) -> Result<Vec<usize>, Error> {
    checked_array(r#type)?;
    Ok(vec![ArrayAddressing::new(r#type.clone())?.element_count()])
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{Layout, StridedLayout, TiledLayout};
    use ryft_mlir::dialects::nvgpu;

    use super::*;

    #[test]
    fn test_build() {
        let context = Context::new();
        let target = Target::new(12, 1).unwrap().with_threads_per_block(64).unwrap();
        let module = build(
            &context,
            &target,
            "mixed_buffers",
            &[(0, ArrayType::scalar(DataType::I32)), (2, ArrayType::new_static(DataType::F32, [2, 3]))],
            &[ArrayType::new_static(DataType::F32, [2, 3])],
            &[],
            &[],
            [2, 1, 1],
            |block, arguments, shared, _descriptors| {
                assert_eq!(arguments.len(), 2);
                assert_eq!(arguments[0].r#type()?.to_string(), "memref<1xi32>");
                assert_eq!(arguments[1].r#type()?.to_string(), "memref<6xf32>");
                assert_eq!(shared.len(), 1);
                assert_eq!(shared[0].r#type()?.to_string(), "memref<6xf32, 3>");
                assert_eq!(block.arguments().count(), 13);
                assert_eq!(block.argument(12)?.as_ref(), shared[0]);
                block.append_operation(gpu::barrier(None, context.unknown_location())?)?;
                Ok(())
            },
        )
        .unwrap();
        assert!(module.verify().unwrap());
        let operation = module.as_operation().unwrap();
        assert_eq!(operation.attribute("mosaic_gpu.arch_major").unwrap().unwrap().to_string(), "12 : i32");
        assert_eq!(operation.attribute("mosaic_gpu.arch_minor").unwrap().unwrap().to_string(), "1 : i32");
        let mut slots = Vec::new();
        let mut pointers = Vec::new();
        operation.walk(WalkOrder::PreOrder, |operation| {
            if let Some(slot) = operation.attribute(KERNEL_ARGUMENT_INDEX_ATTRIBUTE).unwrap() {
                slots.push(slot.to_string());
            }
            if operation.name().as_str() == Ok("llvm.getelementptr") {
                pointers.push(operation.attribute("rawConstantIndices").unwrap().unwrap().to_string());
            }
            WalkResult::Advance
        });
        assert_eq!(slots, vec!["0 : i32", "2 : i32"]);
        assert_eq!(pointers, vec!["array<i32: 0>", "array<i32: 2>"]);
    }

    #[test]
    fn test_build_cluster() {
        let context = Context::new();
        let target = Target::new(9, 0).unwrap().with_blocks_per_cluster(2).unwrap();
        let module = build(
            &context,
            &target,
            "clustered",
            &[],
            &[ArrayType::scalar(DataType::F32)],
            &[],
            &[],
            [4, 1, 1],
            |block, arguments, shared, _| {
                assert_eq!(arguments.len(), 0);
                assert_eq!(block.arguments().count(), 19);
                assert_eq!(block.argument(18)?.as_ref(), shared[0]);
                assert_eq!(block.argument(12)?.r#type()?.to_string(), "index");
                assert_eq!(block.argument(15)?.r#type()?.to_string(), "index");
                Ok(())
            },
        )
        .unwrap();
        assert!(module.verify().unwrap());
        let mut segments = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if operation.name().as_str() == Ok("gpu.launch") {
                segments.push(operation.attribute("operandSegmentSizes").unwrap().unwrap().to_string());
            }
            WalkResult::Advance
        });
        assert_eq!(segments, vec!["array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0>"]);
        assert!(matches!(build(&context, &target, "invalid_cluster", &[], &[], &[], &[], [3, 1, 1],
            |_, _, _, _| Ok(())), Err(Error::Invalid { message })
            if message == "physical grid must be divisible by the cluster dimensions"));
    }

    #[test]
    fn test_build_rejects_invalid_grid() {
        let context = Context::new();
        assert!(matches!(build(&context, &Target::new(9, 0).unwrap(), "empty", &[], &[],
            &[], &[], [0, 1, 1],
            |_, _, _, _| panic!("invalid geometry must be rejected before body construction")),
            Err(Error::Invalid { message })
                if message == "kernel argument count or launch grid exceeds the native ABI"));
    }

    #[test]
    fn test_build_rejects_invalid_argument_slot() {
        let context = Context::new();
        assert!(matches!(build(&context, &Target::new(9, 0).unwrap(), "invalid_slot",
            &[(i32::MAX as usize + 1, ArrayType::scalar(DataType::I32))], &[],
            &[], &[], [1, 1, 1],
            |_, _, _, _| panic!("invalid native slot must be rejected before body construction")),
            Err(Error::Invalid { message })
                if message == "kernel argument count or launch grid exceeds the native ABI"));
    }

    #[test]
    fn test_element_type() {
        let context = Context::new();
        for (data_type, expected) in [
            (DataType::Boolean, "i1"),
            (DataType::U8, "i8"),
            (DataType::F8E4M3FN, "i8"),
            (DataType::F8E8M0FNU, "i8"),
            (DataType::I32, "i32"),
            (DataType::U32, "i32"),
            (DataType::I64, "i64"),
            (DataType::U64, "i64"),
            (DataType::F16, "f16"),
            (DataType::BF16, "bf16"),
            (DataType::F32, "f32"),
            (DataType::F64, "f64"),
        ] {
            assert_eq!(element_type(&context, data_type).unwrap().to_string(), expected);
        }
        assert!(
            matches!(element_type(&context, DataType::Token), Err(Error::Unsupported { operation: "buffer", reason })
            if reason == "unsupported native storage data type `token`")
        );
        for data_type in [DataType::I8, DataType::I16, DataType::U16] {
            assert!(
                matches!(element_type(&context, data_type), Err(Error::Unsupported { operation: "buffer", reason })
                if reason == format!("unsupported native storage data type `{data_type}`"))
            );
        }
    }

    #[test]
    fn test_memref_type() {
        let context = Context::new();
        let r#type = ArrayType::new_static(DataType::F32, [0, 3]);
        assert_eq!(memref_type(&context, &r#type, false).unwrap().to_string(), "memref<0xf32>");
        assert_eq!(memref_type(&context, &r#type, true).unwrap().to_string(), "memref<0xf32, 3>");
        let row_major = ArrayType::new_static(DataType::F32, [2, 3])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![])));
        assert_eq!(memref_type(&context, &row_major, false).unwrap().to_string(), "memref<6xf32>");
        let strided =
            ArrayType::new_static(DataType::F32, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert!(matches!(memref_type(&context, &strided, false), Err(Error::Invalid { message })
            if message == "mosaic GPU requires untiled dense row-major arrays"));
    }

    #[test]
    fn test_serialize() {
        let context = Context::new();
        let module = build(
            &context,
            &Target::new(9, 0).unwrap(),
            "serialization",
            &[],
            &[ArrayType::new_static(DataType::F32, [4])],
            &[],
            &[],
            [1, 1, 1],
            |_, _, _, _| Ok(()),
        )
        .unwrap();
        let bytes = serialize(&module).unwrap();
        assert!(bytes.starts_with(b"ML\xefR"));
        assert_eq!(mosaic_gpu_serde_version(&module), Ok(Some(MOSAIC_GPU_SERDE_VERSION as i64)));
        let parsed = context.parse_module_from_bytes(&bytes).unwrap();
        assert!(parsed.verify().unwrap());
        assert_eq!(parsed.as_operation().unwrap().bytecode_for_version(0).unwrap(), bytes);
        assert_eq!(parsed.to_string(), module.to_string());
        let mut alignments = Vec::new();
        parsed.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if let Some(attributes) = operation.attribute(gpu::WORKGROUP_ATTRIBUTION_ATTRIBUTES_ATTRIBUTE).unwrap() {
                alignments.push(attributes.to_string());
            }
            WalkResult::Advance
        });
        assert_eq!(alignments, vec!["[{llvm.align = 16 : i64}]"]);
    }

    #[test]
    fn test_serialize_shared_alignments() {
        let context = Context::new();
        let module = build(
            &context,
            &Target::new(9, 0).unwrap(),
            "aligned_buffers",
            &[],
            &[ArrayType::new_static(DataType::F32, [32]), ArrayType::new_static(DataType::F16, [128])],
            &[128, 256],
            &[],
            [1, 1, 1],
            |_, _, _, _| Ok(()),
        )
        .unwrap();
        assert!(module.verify().unwrap());
        serialize(&module).unwrap();
        let mut attributes = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if let Some(value) = operation.attribute(gpu::WORKGROUP_ATTRIBUTION_ATTRIBUTES_ATTRIBUTE).unwrap() {
                attributes.push(value.to_string());
            }
            WalkResult::Advance
        });
        assert_eq!(attributes, vec!["[{llvm.align = 128 : i64}, {llvm.align = 256 : i64}]"]);
    }

    #[test]
    fn test_serialize_async_tokens() {
        let context = Context::new();
        let location = context.unknown_location();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let module = build(
            &context,
            &Target::new(9, 0).unwrap(),
            "async_serialization",
            &[(0, r#type.clone())],
            &[r#type],
            &[],
            &[],
            [1, 1, 1],
            |block, arguments, shared, _descriptors| {
                let index =
                    append(block, arith::constant(context.integer_attribute(context.index_type(), 0), location)?)?;
                let token = append(
                    block,
                    nvgpu::device_async_copy(shared[0], &[index], arguments[0], &[index], 1, None, false, location)?,
                )?;
                let group = append(block, nvgpu::device_async_create_group(&[token], location)?)?;
                block.append_operation(nvgpu::device_async_wait(group, Some(0), location)?)?;
                Ok(())
            },
        )
        .unwrap();
        let bytes = serialize(&module).unwrap();
        let rendered = module.to_string();
        assert!(!rendered.contains("!nvgpu.device.async.token"));
        let mut asynchronous_operations = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            let name = operation.name().as_str().unwrap().to_owned();
            if name.starts_with("stable_mosaic_gpu.nvvm.cp.async.") {
                asynchronous_operations.push((name, operation.results().count()));
            }
            WalkResult::Advance
        });
        assert_eq!(
            asynchronous_operations,
            vec![
                ("stable_mosaic_gpu.nvvm.cp.async.shared.global".to_owned(), 0),
                ("stable_mosaic_gpu.nvvm.cp.async.commit.group".to_owned(), 0),
                ("stable_mosaic_gpu.nvvm.cp.async.wait.group".to_owned(), 0),
            ]
        );
        let restored_context = Context::new();
        restored_context.allow_unregistered_dialects();
        for dialect in [
            DialectHandle::arith().unwrap(),
            DialectHandle::func().unwrap(),
            DialectHandle::gpu().unwrap(),
            DialectHandle::llvm().unwrap(),
            DialectHandle::memref().unwrap(),
        ] {
            restored_context.load_dialect(dialect).unwrap();
        }
        let restored = restored_context.parse_module_from_bytes(&bytes).unwrap();
        restored_context.load_dialect(DialectHandle::nvgpu().unwrap()).unwrap();
        let manager = mosaic_gpu_serde_pass_manager(&restored_context, false, None).unwrap();
        assert!(manager.run(&restored.as_operation().unwrap()).is_success());
        assert!(restored.verify().unwrap());
    }
}
