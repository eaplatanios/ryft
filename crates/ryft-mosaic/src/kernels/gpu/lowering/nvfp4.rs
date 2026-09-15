//! Warp-level NVFP4 fragments over explicit packed bytes and signed block scales.

use ryft_core::DataType;
use ryft_mlir::dialects::{arith, llvm, scf};
use ryft_mlir::{Attribute, Block, DetachedBlock, Operation, Type, TypeRef, UnknownLocationRef, Value};

use crate::kernels::gpu::Error;
use crate::kernels::gpu::lowering::{Buffer, KernelValue, Lowering, append, shape};

/// Exact pinned LLVM intrinsic for the native 16-by-8-by-64 E2M1 / unsigned-E4M3 instruction. Mosaic's stable
/// dialect decoder does not reconstruct `mma.block_scale` attributes; the LLVM intrinsic retains the same typed
/// operand/result contract without introducing an assembly-level interface or a second execution path.
const NVFP4_INTRINSIC: &str = "llvm.nvvm.mma.block.scale.m16n8k64.row.col.mxf4nvf4.scale.4x.f32.e2m1.e2m1.f32.ue4m3";

/// Exact pinned ordered-metadata sparse intrinsic, with four packed A and B fragments and one metadata register.
const NVFP4_SPARSE_INTRINSIC: &str =
    "llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k128.row.col.mxf4nvf4.scale.4x.f32.e2m1.e2m1.f32.ue4m3";

impl<'c, 't> Lowering<'c, 't> {
    /// Evaluates complete 16-by-8 tiles using one warp and 64-element contraction fragments. Source types and
    /// geometry are admitted before construction. Signed scale bits move into each matching FP4 block, while
    /// the instruction receives unsigned scale magnitudes. Tensor scaling precedes the final accumulator add.
    pub(super) fn nvfp4(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        inputs: &[Buffer<'c, 't>],
        output: &Buffer<'c, 't>,
        tensor_scale: bool,
    ) -> Result<(), Error> {
        self.nvfp4_product(block, inputs, output, tensor_scale, false)
    }

    /// Validates all ordered pair selectors before evaluating native pair-wise sparse instruction tiles.
    pub(super) fn nvfp4_sparse(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        inputs: &[Buffer<'c, 't>],
        output: &Buffer<'c, 't>,
        tensor_scale: bool,
    ) -> Result<(), Error> {
        let count = shape(&inputs[5].r#type)?.into_iter().product();
        let zero = self.index(block, 0)?;
        let one = self.index(block, 1)?;
        let count = self.index(block, count)?;
        // Every thread scans the same metadata range, so both assertion branches are uniform across the warp.
        let mut body = self.context.block(&[(self.context.index_type().as_ref(), self.location)]);
        let index = body.argument(0)?.as_ref();
        let metadata = self.load(&mut body, &inputs[5], index)?;
        let metadata =
            append(&mut body, arith::extui(metadata, self.context.signless_integer_type(32), self.location)?)?;
        let mut valid = None;
        for code in [0x4, 0x8, 0x9, 0xc, 0xd, 0xe] {
            let code = self.literal(&mut body, DataType::U32, code)?;
            let equal = append(
                &mut body,
                arith::cmpi(metadata, code, arith::IntegerComparisonPredicate::Equal, self.location)?,
            )?;
            valid = Some(match valid {
                Some(previous) => append(&mut body, arith::ori(previous, equal, self.location)?)?,
                None => equal,
            });
        }
        let mut success = self.context.block::<TypeRef<'c, 't>, UnknownLocationRef<'c, 't>>(&[]);
        success.append_operation(scf::r#yield(&[], self.location)?)?;
        let mut failure = self.context.block::<TypeRef<'c, 't>, UnknownLocationRef<'c, 't>>(&[]);
        failure.append_operation(llvm::intr_trap(self.location)?)?;
        failure.append_operation(scf::r#yield(&[], self.location)?)?;
        body.append_operation(scf::r#if(
            valid.unwrap(),
            &[],
            success.try_into()?,
            Some(failure.try_into()?),
            self.location,
        )?)?;
        body.append_operation(scf::r#yield(&[], self.location)?)?;
        block.append_operation(scf::r#for(zero, count, one, &[], false, body.try_into()?, self.location)?)?;
        self.nvfp4_product(block, inputs, output, tensor_scale, true)
    }

    /// Shares fragment packing and final FP32 scaling across the two explicitly selected native contracts.
    fn nvfp4_product(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        inputs: &[Buffer<'c, 't>],
        output: &Buffer<'c, 't>,
        tensor_scale: bool,
        sparse: bool,
    ) -> Result<(), Error> {
        let dimensions = shape(&output.r#type)?;
        let packed = shape(&inputs[1].r#type)?;
        let rows = dimensions[0];
        let columns = dimensions[1];
        let contraction = packed[1] * 2;
        let zero = self.index(block, 0)?;
        let one = self.index(block, 1)?;
        let four = self.index(block, 4)?;
        let eight = self.index(block, 8)?;
        let sixteen = self.index(block, 16)?;
        let instruction_width = if sparse { 128 } else { 64 };
        let sixty_four = self.index(block, 64)?;
        let column_tiles = self.index(block, columns / 8)?;
        let tiles = self.index(block, rows / 16 * (columns / 8))?;
        let chunks = self.index(block, contraction / instruction_width)?;
        let group = append(block, arith::divui(self.thread, four, self.location)?)?;
        let within = append(block, arith::remui(self.thread, four, self.location)?)?;
        let float_type = self.context.float32_type().as_ref();
        let mut tile_body = self.context.block(&[(self.context.index_type().as_ref(), self.location)]);
        let tile = tile_body.argument(0)?.as_ref();
        let tile_row = append(&mut tile_body, arith::divui(tile, column_tiles, self.location)?)?;
        let tile_row = append(&mut tile_body, arith::muli(tile_row, sixteen, self.location)?)?;
        let tile_column = append(&mut tile_body, arith::remui(tile, column_tiles, self.location)?)?;
        let tile_column = append(&mut tile_body, arith::muli(tile_column, eight, self.location)?)?;
        let row = append(&mut tile_body, arith::addi(tile_row, group, self.location)?)?;
        let column = append(&mut tile_body, arith::addi(tile_column, group, self.location)?)?;
        let float_zero = self.literal(&mut tile_body, DataType::F32, 0)?;
        let mut contraction_body = self.context.block(&[
            (self.context.index_type().as_ref(), self.location),
            (float_type, self.location),
            (float_type, self.location),
            (float_type, self.location),
            (float_type, self.location),
        ]);
        let chunk = contraction_body.argument(0)?.as_ref();
        let contraction_base = append(&mut contraction_body, arith::muli(chunk, sixty_four, self.location)?)?;
        let lane_base = append(&mut contraction_body, arith::muli(within, eight, self.location)?)?;
        let base = append(&mut contraction_body, arith::addi(contraction_base, lane_base, self.location)?)?;
        let mut operands = Vec::with_capacity(if sparse { 20 } else { 16 });
        for register in 0..4 {
            let row_offset = self.index(&mut contraction_body, (register % 2) * 8)?;
            let source_row = append(&mut contraction_body, arith::addi(row, row_offset, self.location)?)?;
            let offset = self.index(&mut contraction_body, (register / 2) * 32)?;
            let source_base = append(&mut contraction_body, arith::addi(base, offset, self.location)?)?;
            operands.push(self.nvfp4_fragment(
                &mut contraction_body,
                &inputs[0],
                &inputs[2],
                source_row,
                source_base,
                if sparse { contraction / 2 } else { contraction },
                16,
            )?);
        }
        let base = if sparse {
            let instruction_width = self.index(&mut contraction_body, instruction_width)?;
            let logical_base = append(&mut contraction_body, arith::muli(chunk, instruction_width, self.location)?)?;
            append(&mut contraction_body, arith::addi(logical_base, lane_base, self.location)?)?
        } else {
            base
        };
        for register in 0..if sparse { 4 } else { 2 } {
            let offset = self.index(&mut contraction_body, register * 32)?;
            let source_base = append(&mut contraction_body, arith::addi(base, offset, self.location)?)?;
            operands.push(self.nvfp4_fragment(
                &mut contraction_body,
                &inputs[1],
                &inputs[3],
                column,
                source_base,
                contraction,
                if sparse { 32 } else { 16 },
            )?);
        }
        for index in 1..=4 {
            operands.push(contraction_body.argument(index)?.as_ref());
        }
        let two = self.index(&mut contraction_body, 2)?;
        let scale_row_offset = append(&mut contraction_body, arith::remui(within, two, self.location)?)?;
        let scale_row_offset = append(&mut contraction_body, arith::muli(scale_row_offset, eight, self.location)?)?;
        let scale_row = append(&mut contraction_body, arith::addi(row, scale_row_offset, self.location)?)?;
        if sparse {
            let half = append(&mut contraction_body, arith::divui(within, two, self.location)?)?;
            let eight = self.index(&mut contraction_body, 8)?;
            let half = append(&mut contraction_body, arith::muli(half, eight, self.location)?)?;
            let sixteen = self.index(&mut contraction_body, 16)?;
            let metadata_base = append(&mut contraction_body, arith::muli(chunk, sixteen, self.location)?)?;
            let metadata_base = append(&mut contraction_body, arith::addi(metadata_base, half, self.location)?)?;
            let mut metadata = self.literal(&mut contraction_body, DataType::U32, 0)?;
            for position in 0..8 {
                let offset = self.index(&mut contraction_body, position)?;
                let column = append(&mut contraction_body, arith::addi(metadata_base, offset, self.location)?)?;
                let address = self.nvfp4_address(&mut contraction_body, scale_row, contraction / 8, column)?;
                let code = self.load(&mut contraction_body, &inputs[5], address)?;
                let code = append(
                    &mut contraction_body,
                    arith::extui(code, self.context.signless_integer_type(32), self.location)?,
                )?;
                let shift = self.literal(&mut contraction_body, DataType::U32, position as u64 * 4)?;
                let code = append(&mut contraction_body, arith::shli(code, shift, self.location)?)?;
                metadata = append(&mut contraction_body, arith::ori(metadata, code, self.location)?)?;
            }
            operands.push(metadata);
            operands.push(self.literal(&mut contraction_body, DataType::U32, 0)?);
        }
        let scale_base = append(&mut contraction_body, arith::muli(chunk, four, self.location)?)?;
        let scale_a = self.nvfp4_scales(
            &mut contraction_body,
            &inputs[2],
            scale_row,
            scale_base,
            contraction / if sparse { 32 } else { 16 },
        )?;
        let scale_b = self.nvfp4_scales(
            &mut contraction_body,
            &inputs[3],
            column,
            scale_base,
            contraction / if sparse { 32 } else { 16 },
        )?;
        let selector = append(
            &mut contraction_body,
            arith::constant(self.context.integer_attribute(self.context.signless_integer_type(16), 0), self.location)?,
        )?;
        operands.extend([scale_a, selector, selector, scale_b, selector, selector]);
        let result_type = self.context.llvm_literal_struct_type(&[float_type; 4], false)?.as_ref();
        let result = append(
            &mut contraction_body,
            llvm::call_intrinsic(
                &operands,
                &[],
                result_type,
                self.context
                    .string_attribute(if sparse { NVFP4_SPARSE_INTRINSIC } else { NVFP4_INTRINSIC })
                    .as_ref(),
                None,
                None,
                None,
                None,
                None,
                self.location,
            )?,
        )?;
        let results = (0..4)
            .map(|index| {
                append(
                    &mut contraction_body,
                    llvm::extract_value(
                        result,
                        float_type,
                        self.context.dense_i64_array_attribute(&[index])?.as_ref(),
                        self.location,
                    )?,
                )
            })
            .collect::<Result<Vec<_>, Error>>()?;
        contraction_body.append_operation(scf::r#yield(&results, self.location)?)?;
        let contraction_loop = tile_body.append_operation(scf::r#for(
            zero,
            chunks,
            one,
            &[float_zero; 4],
            false,
            contraction_body.try_into()?,
            self.location,
        )?)?;
        let results = (0..4)
            .map(|index| contraction_loop.result(index).map(|result| result.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let scale = if tensor_scale {
            Some(self.load(&mut tile_body, &inputs[if sparse { 6 } else { 5 }], zero)?)
        } else {
            None
        };
        for (index, mut value) in results.into_iter().enumerate() {
            let row_offset = self.index(&mut tile_body, index / 2 * 8)?;
            let result_row = append(&mut tile_body, arith::addi(row, row_offset, self.location)?)?;
            let two = self.index(&mut tile_body, 2)?;
            let column_offset = append(&mut tile_body, arith::muli(within, two, self.location)?)?;
            let column_offset = append(&mut tile_body, arith::addi(tile_column, column_offset, self.location)?)?;
            let odd = self.index(&mut tile_body, index % 2)?;
            let result_column = append(&mut tile_body, arith::addi(column_offset, odd, self.location)?)?;
            let address = self.nvfp4_address(&mut tile_body, result_row, columns, result_column)?;
            if let Some(scale) = scale {
                value = append(&mut tile_body, arith::mulf(value, scale, self.location)?)?;
            }
            let accumulator = self.load(&mut tile_body, &inputs[4], address)?;
            let value = append(&mut tile_body, arith::addf(value, accumulator, self.location)?)?;
            self.store(&mut tile_body, output, address, value)?;
        }
        tile_body.append_operation(scf::r#yield(&[], self.location)?)?;
        block.append_operation(scf::r#for(zero, tiles, one, &[], false, tile_body.try_into()?, self.location)?)?;
        self.barrier(block)
    }

    /// Packs eight FP4 elements and folds their shared signed block-scale bit into the two nibbles of each byte.
    fn nvfp4_fragment(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        packed: &Buffer<'c, 't>,
        scales: &Buffer<'c, 't>,
        row: KernelValue<'c, 't>,
        base: KernelValue<'c, 't>,
        contraction: usize,
        scale_span: usize,
    ) -> Result<KernelValue<'c, 't>, Error> {
        let sixteen = self.index(block, scale_span)?;
        let group = append(block, arith::divui(base, sixteen, self.location)?)?;
        let scale_address = self.nvfp4_address(block, row, contraction / scale_span, group)?;
        let scale = self.load(block, scales, scale_address)?;
        let scale = append(block, arith::extui(scale, self.context.signless_integer_type(32), self.location)?)?;
        let seven = self.literal(block, DataType::U32, 7)?;
        let sign = append(block, arith::shrui(scale, seven, self.location)?)?;
        let mask = self.literal(block, DataType::U32, 0x88)?;
        let sign = append(block, arith::muli(sign, mask, self.location)?)?;
        let two = self.index(block, 2)?;
        let byte_base = append(block, arith::divui(base, two, self.location)?)?;
        let mut result = self.literal(block, DataType::U32, 0)?;
        for byte in 0..4 {
            let offset = self.index(block, byte)?;
            let column = append(block, arith::addi(byte_base, offset, self.location)?)?;
            let address = self.nvfp4_address(block, row, contraction / 2, column)?;
            let value = self.load(block, packed, address)?;
            let value = append(block, arith::extui(value, self.context.signless_integer_type(32), self.location)?)?;
            let value = append(block, arith::xori(value, sign, self.location)?)?;
            let shift = self.literal(block, DataType::U32, byte as u64 * 8)?;
            let value = append(block, arith::shli(value, shift, self.location)?)?;
            result = append(block, arith::ori(result, value, self.location)?)?;
        }
        Ok(result)
    }

    /// Packs four nonnegative E4M3 magnitudes in contraction-block order for the hardware scale operand.
    fn nvfp4_scales(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        scales: &Buffer<'c, 't>,
        row: KernelValue<'c, 't>,
        base: KernelValue<'c, 't>,
        stride: usize,
    ) -> Result<KernelValue<'c, 't>, Error> {
        let mut result = self.literal(block, DataType::U32, 0)?;
        let mask = self.literal(block, DataType::U32, 0x7f)?;
        for byte in 0..4 {
            let offset = self.index(block, byte)?;
            let column = append(block, arith::addi(base, offset, self.location)?)?;
            let address = self.nvfp4_address(block, row, stride, column)?;
            let value = self.load(block, scales, address)?;
            let value = append(block, arith::extui(value, self.context.signless_integer_type(32), self.location)?)?;
            let value = append(block, arith::andi(value, mask, self.location)?)?;
            let shift = self.literal(block, DataType::U32, byte as u64 * 8)?;
            let value = append(block, arith::shli(value, shift, self.location)?)?;
            result = append(block, arith::ori(result, value, self.location)?)?;
        }
        Ok(result)
    }

    /// Forms a checked row-major physical byte or scalar index.
    fn nvfp4_address(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        row: KernelValue<'c, 't>,
        stride: usize,
        column: KernelValue<'c, 't>,
    ) -> Result<KernelValue<'c, 't>, Error> {
        let stride = self.index(block, stride)?;
        let offset = append(block, arith::muli(row, stride, self.location)?)?;
        append(block, arith::addi(offset, column, self.location)?)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::kernels::whole_array_parameter;
    use ryft_core::kernels::{
        Grid, KernelCallOperation, KernelDefinition, KernelOperation, KernelParameterAccess, KernelSchedule,
        VerifiedKernel,
    };
    use ryft_core::{ArrayType, Context as CoreContext, ReferenceRead, ReferenceWrite};
    use ryft_mlir::dialects::mosaic::gpu::mosaic_gpu_serde_pass_manager;
    use ryft_mlir::{Context, DialectHandle, WalkOrder, WalkResult};

    use crate::kernels::gpu::lowering::module;
    use crate::kernels::gpu::{Compiler, GpuOperation, Options, Target};

    use super::*;

    /// Builds the actual typed target operation with full input/output reference boundaries.
    fn definition(tensor_scale: bool, sparse: bool) -> KernelDefinition<GpuOperation> {
        let mut input_types = vec![
            ArrayType::new_static(DataType::U8, [16, 32]),
            ArrayType::new_static(DataType::U8, [8, if sparse { 64 } else { 32 }]),
            ArrayType::new_static(DataType::F8E4M3FN, [16, 4]),
            ArrayType::new_static(DataType::F8E4M3FN, [8, 4]),
            ArrayType::new_static(DataType::F32, [16, 8]),
        ];
        if sparse {
            input_types.push(ArrayType::new_static(DataType::U8, [16, 16]));
        }
        if tensor_scale {
            input_types.push(ArrayType::scalar(DataType::F32));
        }
        let mut parameters = input_types
            .into_iter()
            .map(|r#type| whole_array_parameter(r#type, KernelParameterAccess::ReadOnly).unwrap())
            .collect::<Vec<_>>();
        parameters.push(
            whole_array_parameter(ArrayType::new_static(DataType::F32, [16, 8]), KernelParameterAccess::WriteOnly)
                .unwrap(),
        );
        let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), parameters).unwrap();
        KernelDefinition::trace(call, |(references, _)| {
            let values = references[..references.len() - 1]
                .iter()
                .map(|reference| reference.read())
                .collect::<Result<Vec<_>, _>>()?;
            let result = references[0]
                .context()
                .bind(
                    KernelOperation::Extension(if sparse {
                        GpuOperation::Nvfp4Sparse { tensor_scale }
                    } else {
                        GpuOperation::Nvfp4 { tensor_scale }
                    }),
                    vec![],
                    &values,
                )?
                .remove(0);
            references.last().unwrap().write(&result)
        })
        .unwrap()
    }

    #[test]
    fn test_lowering_nvfp4() {
        let context = Context::new();
        let definition = definition(true, false);
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let module = Compiler
            .module(&context, &kernel, &Target::new(12, 1).unwrap(), &Options::default(), &KernelSchedule::default())
            .unwrap();
        assert!(module.verify().unwrap());
        let mut instructions = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if operation.name().as_str() == Ok("llvm.call_intrinsic") {
                assert_eq!(
                    operation.attribute("intrin").unwrap().unwrap().to_string(),
                    format!("\"{NVFP4_INTRINSIC}\"")
                );
                instructions.push((
                    operation
                        .operand_values()
                        .map(|value| value.unwrap().r#type().unwrap().to_string())
                        .collect::<Vec<_>>(),
                    operation.result_type(0).unwrap().to_string(),
                ));
            }
            WalkResult::Advance
        });
        assert_eq!(
            instructions,
            vec![(
                vec![
                    "i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32", "f32", "f32", "i32", "i16", "i16", "i32",
                    "i16", "i16"
                ]
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>(),
                "!llvm.struct<(f32, f32, f32, f32)>".to_owned()
            )]
        );
    }

    #[test]
    fn test_lowering_nvfp4_sparse() {
        let context = Context::new();
        let definition = definition(true, true);
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let source = Compiler
            .module(&context, &kernel, &Target::new(12, 1).unwrap(), &Options::default(), &KernelSchedule::default())
            .unwrap();
        assert!(source.verify().unwrap());
        let mut ordered = Vec::new();
        source.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if operation.name().as_str() == Ok("llvm.intr.trap") {
                ordered.push("trap".to_owned());
            }
            if operation.name().as_str() == Ok("llvm.call_intrinsic") {
                assert_eq!(
                    operation.attribute("intrin").unwrap().unwrap().to_string(),
                    format!("\"{NVFP4_SPARSE_INTRINSIC}\"")
                );
                assert_eq!(
                    operation
                        .operand_values()
                        .map(|value| value.unwrap().r#type().unwrap().to_string())
                        .collect::<Vec<_>>(),
                    [
                        "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32", "f32", "f32", "i32",
                        "i32", "i32", "i16", "i16", "i32", "i16", "i16"
                    ]
                    .map(str::to_owned)
                );
                assert_eq!(operation.result_type(0).unwrap().to_string(), "!llvm.struct<(f32, f32, f32, f32)>");
                ordered.push("sparse".to_owned());
            }
            WalkResult::Advance
        });
        assert_eq!(ordered, ["trap", "sparse"]);
    }

    #[test]
    fn test_lowering_nvfp4_serialization() {
        let context = Context::new();
        for sparse in [false, true] {
            let definition = definition(false, sparse);
            let kernel = VerifiedKernel::new(&definition, 1).unwrap();
            let source = Compiler
                .module(
                    &context,
                    &kernel,
                    &Target::new(12, 1).unwrap(),
                    &Options::default(),
                    &KernelSchedule::default(),
                )
                .unwrap();
            let bytes = module::serialize(&source).unwrap();
            let restored_context = Context::new();
            restored_context.allow_unregistered_dialects();
            for dialect in [
                DialectHandle::arith().unwrap(),
                DialectHandle::func().unwrap(),
                DialectHandle::gpu().unwrap(),
                DialectHandle::llvm().unwrap(),
                DialectHandle::memref().unwrap(),
                DialectHandle::scf().unwrap(),
            ] {
                restored_context.load_dialect(dialect).unwrap();
            }
            let restored = restored_context.parse_module_from_bytes(&bytes).unwrap();
            restored_context.load_dialect(DialectHandle::nvvm().unwrap()).unwrap();
            let manager = mosaic_gpu_serde_pass_manager(&restored_context, false, None).unwrap();
            assert!(manager.run(&restored.as_operation().unwrap()).is_success());
            assert!(restored.verify().unwrap());
        }
    }
}
