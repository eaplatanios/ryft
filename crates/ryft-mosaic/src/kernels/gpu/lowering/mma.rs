//! Hopper WGMMA lowering with explicit FP32 accumulation and 32-byte-swizzled shared-memory operands.
//!
//! Operand packing, descriptor fields, and swizzles follow JAX commit `a7606f995e1a92707cbeb257e487fa53e7abe84b`,
//! `jax/experimental/mosaic/gpu/{mma_utils,fragmented_array}.py`. Register ownership follows LLVM commit
//! `979b722373384babff6ea2befe17c5c1c93b64bd`, `mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp`.
//! The [PTX instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-mma) defines the
//! operand shapes, descriptor encoding, asynchronous groups, and accumulator register distribution.
//! These are physical transports of canonical array values, not additional semantic arrays. Each auxiliary allocation
//! must have 256-byte base alignment so the 32-byte swizzle's descriptor base-offset field is zero.

use ryft_core::{
    ArrayReferenceView, ArraySliceAxis, ArrayType, DataType, DotOperation, Operation as CoreOperation,
    ScaledDotOperation,
};
use ryft_mlir::dialects::{arith, llvm, memref, nvvm};
use ryft_mlir::{Attribute, Block, DetachedBlock, Type};

use crate::kernels::gpu::Error;
use crate::kernels::gpu::lowering::{Buffer, KernelValue, Lowering, append, shape};
use crate::kernels::gpu::synchronization::SynchronizationEvent;

/// Checks the exact first WGMMA geometry and returns the two flat shared transport allocation types. The native
/// caller must provide one complete 128-thread warpgroup and align both allocations to 256 bytes. Plain half-type
/// contractions are not silently widened: the canonical operation must explicitly request FP32 accumulation.
pub(super) fn requirements(
    operation: &DotOperation,
    inputs: &[ArrayType],
    output: &ArrayType,
) -> Result<[ArrayType; 2], Error> {
    let unsupported = |reason: &str| Error::Unsupported { operation: "dot", reason: reason.to_owned() };
    if inputs.len() != 2
        || operation.dimensions() != DotOperation::matmul().dimensions()
        || operation.accumulation_type() != Some(DataType::F32)
        || !matches!(inputs[0].data_type(), DataType::F16 | DataType::BF16)
        || inputs[0].data_type() != inputs[1].data_type()
    {
        return Err(unsupported("wgmma requires ordinary F16 or BF16 matmul with explicit F32 accumulation"));
    }
    if operation.infer_output_types(inputs, &[])?.as_slice() != [output.clone()] {
        return Err(unsupported("wgmma result type differs from canonical dot inference"));
    }
    let left = shape(&inputs[0])?;
    let right = shape(&inputs[1])?;
    if left.len() != 2
        || right.len() != 2
        || left[0] != 64
        || right[0] != left[1]
        || left[1] == 0
        || left[1] % 16 != 0
        || right[1] == 0
        || right[1] > 256
        || right[1] % 8 != 0
    {
        return Err(unsupported("wgmma requires M=64, N divisible by 8 in [8, 256], and positive K divisible by 16"));
    }
    // Each descriptor stride is 16*K bytes; the encoding retains exactly bits [4,18).
    if left[1] >= 16_384 {
        return Err(unsupported("wgmma shared-memory stride exceeds the descriptor encoding"));
    }
    Ok([
        ArrayType::new_static(inputs[0].data_type(), [64 * left[1]]),
        ArrayType::new_static(inputs[1].data_type(), [left[1] * right[1]]),
    ])
}

/// Checks canonical scaled-dot inference and the BF16 operand/scale subset with explicit F32 accumulation.
/// Scale expansion is fused into shared-memory packing; the canonical BF16 product rounding is retained.
pub(super) fn scaled_requirements(
    operation: &ScaledDotOperation,
    inputs: &[ArrayType],
    output: &ArrayType,
) -> Result<[ArrayType; 2], Error> {
    let inferred = operation.infer_output_types(inputs, &[])?;
    if operation.dimensions() != DotOperation::matmul().dimensions()
        || operation.preferred_element_type() != DataType::F32
        || inputs.iter().any(|input| input.data_type() != DataType::BF16)
    {
        return Err(Error::Unsupported {
            operation: "scaled_dot",
            reason: "wgmma scaled dot requires ordinary BF16 operands and scales with F32 accumulation".to_owned(),
        });
    }
    if inferred.as_slice() != [output.clone()] {
        return Err(Error::Unsupported {
            operation: "scaled_dot",
            reason: "wgmma scaled-dot result differs from canonical inference".to_owned(),
        });
    }
    requirements(&DotOperation::matmul().with_accumulation_type(DataType::F32), &inputs[..2], output)
}

impl<'c, 't> Lowering<'c, 't> {
    /// Packs operands into K-fastest 8-by-16 atoms, issues bounded committed WGMMA groups, and publishes the FP32
    /// accumulator only after a final wait for all groups. `scratch` contains two admission-sized, 256-byte-aligned
    /// shared memrefs. `stages` bounds outstanding committed groups and must be between one and eight. Each optional
    /// BF16 scale is expanded along its operand's contracting dimension and multiplied before transport packing.
    pub(super) fn wgmma(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        inputs: &[Buffer<'c, 't>],
        output: &Buffer<'c, 't>,
        scratch: &[KernelValue<'c, 't>; 2],
        stages: usize,
        scales: [Option<&Buffer<'c, 't>>; 2],
    ) -> Result<(), Error> {
        if !(1..=8).contains(&stages) {
            return Err(Error::Unsupported { operation: "dot", reason: "wgmma stages must be in [1, 8]".to_owned() });
        }
        let left = shape(&inputs[0].r#type)?;
        let right = shape(&inputs[1].r#type)?;
        let contraction = left[1];
        let columns = right[1];
        for (operand, source) in inputs.iter().enumerate() {
            let scratch = scratch[operand];
            let source_columns = if operand == 0 { contraction } else { columns };
            self.distributed(block, shape(&source.r#type)?.iter().product(), |lowering, body, index| {
                let width = lowering.index(body, source_columns)?;
                let row = append(body, arith::divui(index, width, lowering.location)?)?;
                let column = append(body, arith::remui(index, width, lowering.location)?)?;
                let (major, reduction) = if operand == 0 { (row, column) } else { (column, row) };
                let swizzled = lowering.mma_swizzled_index(body, major, reduction, contraction, 2)?;
                let mut value = lowering.load(body, source, index)?;
                if let Some(scale) = scales[operand] {
                    let scale_shape = shape(&scale.r#type)?;
                    let scale_width = lowering.index(body, scale_shape[1])?;
                    let block_size = contraction / scale_shape[if operand == 0 { 1 } else { 0 }];
                    let block_size = lowering.index(body, block_size)?;
                    let (scale_row, scale_column) = if operand == 0 {
                        (row, append(body, arith::divui(column, block_size, lowering.location)?)?)
                    } else {
                        (append(body, arith::divui(row, block_size, lowering.location)?)?, column)
                    };
                    let scale_start = append(body, arith::muli(scale_row, scale_width, lowering.location)?)?;
                    let scale_index = append(body, arith::addi(scale_start, scale_column, lowering.location)?)?;
                    let scale_value = lowering.load(body, scale, scale_index)?;
                    // A BF16 product is exact in F32 before its single canonical BF16 rounding.
                    let float = lowering.context.float32_type();
                    let element = append(body, arith::extf(value, float, lowering.location)?)?;
                    let scale_value = append(body, arith::extf(scale_value, float, lowering.location)?)?;
                    let product = append(body, arith::mulf(element, scale_value, lowering.location)?)?;
                    value = append(body, arith::truncf(product, lowering.context.bfloat16_type(), lowering.location)?)?;
                }
                body.append_operation(memref::store(value, scratch, &[swizzled], false, None, lowering.location)?)?;
                Ok(())
            })?;
        }
        block.append_operation(nvvm::fence_proxy(
            &[],
            &[],
            &[
                ("kind", self.context.nvvm_proxy_kind_attribute(nvvm::ProxyKind::AsyncShared)?.as_ref()),
                ("space", self.context.nvvm_shared_space_attribute(nvvm::SharedSpace::Cta)?.as_ref()),
            ],
            false,
            self.location,
        )?)?;
        self.wgmma_event(SynchronizationEvent::AsyncProxyFence)?;
        self.barrier(block)?;
        let descriptors = [
            self.wgmma_descriptor(block, scratch[0], contraction)?,
            self.wgmma_descriptor(block, scratch[1], contraction)?,
        ];
        let float = self.context.float32_type();
        let accumulator_type = self.context.llvm_literal_struct_type(&vec![float; columns / 2], false)?;
        let mut accumulator = append(block, llvm::undef(accumulator_type, self.location)?)?;
        let zero = append(block, arith::constant(self.context.float_attribute(float, 0.0), self.location)?)?;
        for register in 0..columns / 2 {
            accumulator = append(
                block,
                llvm::insert_value(
                    accumulator,
                    zero,
                    accumulator_type.as_ref(),
                    self.context.dense_i64_array_attribute(&[register as i64])?.as_ref(),
                    self.location,
                )?,
            )?;
        }
        block.append_operation(nvvm::wgmma_fence_aligned(&[], &[], &[], false, self.location)?)?;
        self.wgmma_event(SynchronizationEvent::WgmmaFence { accumulator: output.owner })?;
        let input_type = match inputs[0].r#type.data_type() {
            DataType::F16 => nvvm::WgmmaType::F16,
            DataType::BF16 => nvvm::WgmmaType::Bf16,
            _ => unreachable!(),
        };
        for group in 0..contraction / 16 {
            // Consecutive K atoms are 256 bytes apart; the descriptor encodes addresses in units of 16 bytes.
            let offset = self.wgmma_integer(block, (group * 16) as u64)?;
            let left = append(block, arith::addi(descriptors[0], offset, self.location)?)?;
            let right = append(block, arith::addi(descriptors[1], offset, self.location)?)?;
            accumulator = append(
                block,
                nvvm::wgmma_mma_async(
                    &[accumulator, left, right],
                    &[accumulator_type.as_ref()],
                    &[
                        (
                            "shape",
                            self.context
                                .nvvm_mma_shape_attribute(nvvm::MmaShape { m: 64, n: columns as i32, k: 16 })?
                                .as_ref(),
                        ),
                        ("typeA", self.context.nvvm_wgmma_type_attribute(input_type)?.as_ref()),
                        ("typeB", self.context.nvvm_wgmma_type_attribute(input_type)?.as_ref()),
                        ("typeD", self.context.nvvm_wgmma_type_attribute(nvvm::WgmmaType::F32)?.as_ref()),
                        ("scaleD", self.context.nvvm_wgmma_scale_out_attribute(nvvm::WgmmaScaleOut::One)?.as_ref()),
                        ("scaleA", self.context.nvvm_wgmma_scale_in_attribute(nvvm::WgmmaScaleIn::One)?.as_ref()),
                        ("scaleB", self.context.nvvm_wgmma_scale_in_attribute(nvvm::WgmmaScaleIn::One)?.as_ref()),
                        ("layoutA", self.context.nvvm_mma_layout_attribute(nvvm::MmaLayout::Row)?.as_ref()),
                        ("layoutB", self.context.nvvm_mma_layout_attribute(nvvm::MmaLayout::Col)?.as_ref()),
                    ],
                    false,
                    self.location,
                )?,
            )?;
            self.wgmma_event(SynchronizationEvent::WgmmaIssue {
                accumulator: output.owner,
                left: (
                    inputs[0].owner,
                    ArrayReferenceView::Slice {
                        axes: vec![ArraySliceAxis::new(0, 64, 1), ArraySliceAxis::new(group * 16, 16, 1)],
                    },
                ),
                right: (
                    inputs[1].owner,
                    ArrayReferenceView::Slice {
                        axes: vec![ArraySliceAxis::new(group * 16, 16, 1), ArraySliceAxis::new(0, columns, 1)],
                    },
                ),
            })?;
            block.append_operation(nvvm::wgmma_commit_group_sync_aligned(&[], &[], &[], false, self.location)?)?;
            self.wgmma_event(SynchronizationEvent::WgmmaCommit)?;
            if group + 1 >= stages {
                self.wgmma_wait(block, stages - 1)?;
            }
        }
        if stages != 1 {
            self.wgmma_wait(block, 0)?;
        }
        let warp_size = self.index(block, 32)?;
        let four = self.index(block, 4)?;
        let sixteen = self.index(block, 16)?;
        let two = self.index(block, 2)?;
        let warp = append(block, arith::divui(self.thread, warp_size, self.location)?)?;
        let lane = append(block, arith::remui(self.thread, warp_size, self.location)?)?;
        let lane_row = append(block, arith::divui(lane, four, self.location)?)?;
        let warp_row = append(block, arith::muli(warp, sixteen, self.location)?)?;
        let row_base = append(block, arith::addi(warp_row, lane_row, self.location)?)?;
        let lane_column = append(block, arith::remui(lane, four, self.location)?)?;
        let column_base = append(block, arith::muli(lane_column, two, self.location)?)?;
        let width = self.index(block, columns)?;
        for register in 0..columns / 2 {
            let row_offset = self.index(block, (register / 2 % 2) * 8)?;
            let column_offset = self.index(block, (register / 4) * 8 + register % 2)?;
            let row = append(block, arith::addi(row_base, row_offset, self.location)?)?;
            let column = append(block, arith::addi(column_base, column_offset, self.location)?)?;
            let row_start = append(block, arith::muli(row, width, self.location)?)?;
            let index = append(block, arith::addi(row_start, column, self.location)?)?;
            let value = append(
                block,
                llvm::extract_value(
                    accumulator,
                    float.as_ref(),
                    self.context.dense_i64_array_attribute(&[register as i64])?.as_ref(),
                    self.location,
                )?,
            )?;
            self.store(block, output, index, value)?;
        }
        self.barrier(block)
    }

    /// Encodes the K-fastest 8-by-16 atom layout with a zero leading offset and a 32-byte swizzle.
    fn wgmma_descriptor(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        buffer: KernelValue<'c, 't>,
        contraction: usize,
    ) -> Result<KernelValue<'c, 't>, Error> {
        let address = self.mma_descriptor_address(block, buffer)?;
        let fields = self.wgmma_integer(block, ((contraction as u64) << 32) | (3 << 62))?;
        append(block, arith::ori(address, fields, self.location)?)
    }

    /// Maps an operand coordinate into 8-row, 32-byte-row atoms with the native 32-byte swizzle.
    pub(super) fn mma_swizzled_index(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        major: KernelValue<'c, 't>,
        reduction: KernelValue<'c, 't>,
        contraction: usize,
        element_bytes: usize,
    ) -> Result<KernelValue<'c, 't>, Error> {
        let eight = self.index(block, 8)?;
        let row_elements = self.index(block, 32 / element_bytes)?;
        let atom_elements = self.index(block, 256 / element_bytes)?;
        let reduction_atoms = self.index(block, contraction / (32 / element_bytes))?;
        let major_atom = append(block, arith::divui(major, eight, self.location)?)?;
        let reduction_atom = append(block, arith::divui(reduction, row_elements, self.location)?)?;
        let atom_row = append(block, arith::muli(major_atom, reduction_atoms, self.location)?)?;
        let atom = append(block, arith::addi(atom_row, reduction_atom, self.location)?)?;
        let atom_start = append(block, arith::muli(atom, atom_elements, self.location)?)?;
        let major_lane = append(block, arith::remui(major, eight, self.location)?)?;
        let reduction_lane = append(block, arith::remui(reduction, row_elements, self.location)?)?;
        let row_offset = append(block, arith::muli(major_lane, row_elements, self.location)?)?;
        let local = append(block, arith::addi(row_offset, reduction_lane, self.location)?)?;
        let packed = append(block, arith::addi(atom_start, local, self.location)?)?;
        // XOR the 16-byte sector with bit 7 of the byte address.
        let group_elements = self.index(block, 128 / element_bytes)?;
        let sector_elements = self.index(block, 16 / element_bytes)?;
        let two = self.index(block, 2)?;
        let group = append(block, arith::divui(packed, group_elements, self.location)?)?;
        let parity = append(block, arith::remui(group, two, self.location)?)?;
        let bits = append(block, arith::muli(parity, sector_elements, self.location)?)?;
        append(block, arith::xori(packed, bits, self.location)?)
    }

    /// Encodes a shared memref's aligned address in the common matrix descriptor's 16-byte units.
    pub(super) fn mma_descriptor_address(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        buffer: KernelValue<'c, 't>,
    ) -> Result<KernelValue<'c, 't>, Error> {
        let pointer = append(block, memref::extract_aligned_pointer_as_index(buffer, self.location)?)?;
        let pointer =
            append(block, arith::index_castui(pointer, self.context.signless_integer_type(64), self.location)?)?;
        let mask = self.wgmma_integer(block, 0x3ffff)?;
        let shift = self.wgmma_integer(block, 4)?;
        let address = append(block, arith::andi(pointer, mask, self.location)?)?;
        append(block, arith::shrui(address, shift, self.location)?)
    }

    /// Emits the exact 64-bit descriptor encoding, including its high swizzle bits.
    fn wgmma_integer(&self, block: &mut DetachedBlock<'c, 't>, value: u64) -> Result<KernelValue<'c, 't>, Error> {
        append(
            block,
            arith::constant(
                self.context.integer_attribute(self.context.signless_integer_type(64), value as i64),
                self.location,
            )?,
        )
    }

    /// Waits until at most `remaining` previously committed WGMMA groups remain outstanding.
    fn wgmma_wait(&mut self, block: &mut DetachedBlock<'c, 't>, remaining: usize) -> Result<(), Error> {
        block.append_operation(nvvm::wgmma_wait_group_sync_aligned(
            &[],
            &[],
            &[(
                "group",
                self.context.integer_attribute(self.context.signless_integer_type(64), remaining as i64).as_ref(),
            )],
            false,
            self.location,
        )?)?;
        self.wgmma_event(SynchronizationEvent::WgmmaWait { remaining })
    }

    /// Records the native collective for the actual complete warpgroup using canonical operand owners.
    fn wgmma_event(&mut self, event: SynchronizationEvent) -> Result<(), Error> {
        if let Some(plans) = &self.synchronization {
            for thread in 0..plans[0].participants().get() {
                self.record_synchronization(None, thread, event.clone())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use pretty_assertions::assert_eq;

    use ryft_core::kernels::{
        Grid, KernelCallOperation, KernelDefinition, KernelParameterAccess, KernelSchedule, VerifiedKernel,
        whole_array_parameter,
    };
    use ryft_core::{ArrayIrOperation, ArrayOperation, Context as CoreContext, ReferenceRead, ReferenceWrite};
    use ryft_mlir::{Context, Operation, Value, WalkOrder, WalkResult};

    use crate::kernels::gpu::{Compiler, Mma, Options, Target};

    use super::*;

    #[test]
    fn test_requirements() {
        let operation = DotOperation::matmul().with_accumulation_type(DataType::F32);
        for data_type in [DataType::F16, DataType::BF16] {
            let inputs = [ArrayType::new_static(data_type, [64, 32]), ArrayType::new_static(data_type, [32, 24])];
            let output = ArrayType::new_static(DataType::F32, [64, 24]);
            assert_eq!(
                requirements(&operation, &inputs, &output).unwrap(),
                [ArrayType::new_static(data_type, [2048]), ArrayType::new_static(data_type, [768]),],
            );
        }
    }

    #[test]
    fn test_requirements_rejects_incompatible_contracts() {
        let inputs = [ArrayType::new_static(DataType::BF16, [64, 16]), ArrayType::new_static(DataType::BF16, [16, 8])];
        assert!(
            matches!(requirements(&DotOperation::matmul(), &inputs, &ArrayType::new_static(DataType::BF16, [64, 8])),
            Err(Error::Unsupported { operation: "dot", reason })
            if reason == "wgmma requires ordinary F16 or BF16 matmul with explicit F32 accumulation")
        );
        let operation = DotOperation::matmul().with_accumulation_type(DataType::F32);
        for (rows, contraction, columns) in [(32, 16, 8), (64, 8, 8), (64, 16, 7), (64, 16, 264), (64, 0, 8)] {
            let inputs = [
                ArrayType::new_static(DataType::F16, [rows, contraction]),
                ArrayType::new_static(DataType::F16, [contraction, columns]),
            ];
            assert!(
                matches!(requirements(&operation, &inputs, &ArrayType::new_static(DataType::F32, [rows, columns])),
                Err(Error::Unsupported { operation: "dot", reason })
                if reason == "wgmma requires M=64, N divisible by 8 in [8, 256], and positive K divisible by 16")
            );
        }
    }

    #[test]
    fn test_scaled_requirements() {
        let left = ArrayType::new_static(DataType::BF16, [64, 32]);
        let right = ArrayType::new_static(DataType::BF16, [32, 8]);
        let left_scale = ArrayType::new_static(DataType::BF16, [64, 2]);
        let right_scale = ArrayType::new_static(DataType::BF16, [4, 8]);
        let output = ArrayType::new_static(DataType::F32, [64, 8]);
        for (has_left, has_right, inputs) in [
            (false, false, vec![left.clone(), right.clone()]),
            (true, false, vec![left.clone(), right.clone(), left_scale.clone()]),
            (false, true, vec![left.clone(), right.clone(), right_scale.clone()]),
            (true, true, vec![left.clone(), right.clone(), left_scale, right_scale]),
        ] {
            let operation = ScaledDotOperation::new(
                DotOperation::matmul().dimensions().clone(),
                DataType::F32,
                has_left,
                has_right,
            );
            assert_eq!(
                scaled_requirements(&operation, &inputs, &output).unwrap(),
                [ArrayType::new_static(DataType::BF16, [2048]), ArrayType::new_static(DataType::BF16, [256]),]
            );
        }
    }

    #[test]
    fn test_scaled_requirements_rejects_other_element_contracts() {
        let operation =
            ScaledDotOperation::new(DotOperation::matmul().dimensions().clone(), DataType::F32, true, false);
        let inputs = [
            ArrayType::new_static(DataType::BF16, [64, 32]),
            ArrayType::new_static(DataType::BF16, [32, 8]),
            ArrayType::new_static(DataType::F32, [64, 2]),
        ];
        assert!(matches!(scaled_requirements(&operation, &inputs, &ArrayType::new_static(DataType::F32, [64, 8])),
            Err(Error::Unsupported { operation: "scaled_dot", reason })
            if reason == "wgmma scaled dot requires ordinary BF16 operands and scales with F32 accumulation"));
    }

    #[test]
    fn test_lowering_wgmma() {
        for (data_type, stages) in [(DataType::F16, 1), (DataType::BF16, 1), (DataType::F16, 2), (DataType::F16, 8)] {
            let parameters = [
                whole_array_parameter(ArrayType::new_static(data_type, [64, 32]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(ArrayType::new_static(data_type, [32, 8]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(ArrayType::new_static(DataType::F32, [64, 8]), KernelParameterAccess::WriteOnly)
                    .unwrap(),
            ];
            let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), parameters.to_vec()).unwrap();
            let definition: KernelDefinition = KernelDefinition::trace(call, |(references, _)| {
                let left = references[0].read()?;
                let right = references[1].read()?;
                let result = left
                    .context()
                    .bind(
                        ArrayIrOperation::Array(ArrayOperation::Dot(
                            DotOperation::matmul().with_accumulation_type(DataType::F32),
                        )),
                        vec![],
                        &[left.clone(), right],
                    )?
                    .remove(0);
                references[2].write(&result)
            })
            .unwrap();
            let context = Context::new();
            let target = Target::new(9, 0).unwrap().with_threads_per_block(128).unwrap();
            let kernel = VerifiedKernel::new(&definition, 1).unwrap();
            let module = Compiler
                .module(
                    &context,
                    &kernel,
                    &target,
                    &Options::default().with_mma(Mma::Wgmma),
                    &KernelSchedule::default().with_pipeline_stages(NonZeroUsize::new(stages).unwrap()),
                )
                .unwrap();
            assert!(module.verify().unwrap());
            let mut operations = Vec::new();
            let mut shapes = Vec::new();
            let mut layouts = Vec::new();
            let mut groups = Vec::new();
            module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
                let name = operation.name().to_string();
                if name.starts_with("nvvm.wgmma") || name == "nvvm.fence.proxy" {
                    operations.push(name.clone());
                }
                if name == "nvvm.wgmma.mma_async" {
                    shapes.push(operation.attribute("shape").unwrap().unwrap().to_string());
                    layouts.push((
                        operation.attribute("layoutA").unwrap().unwrap().to_string(),
                        operation.attribute("layoutB").unwrap().unwrap().to_string(),
                    ));
                }
                if name == "nvvm.wgmma.wait.group.sync.aligned" {
                    groups.push(operation.attribute("group").unwrap().unwrap().to_string());
                }
                WalkResult::Advance
            });
            let expected = match stages {
                1 => vec![
                    "nvvm.fence.proxy",
                    "nvvm.wgmma.fence.aligned",
                    "nvvm.wgmma.mma_async",
                    "nvvm.wgmma.commit.group.sync.aligned",
                    "nvvm.wgmma.wait.group.sync.aligned",
                    "nvvm.wgmma.mma_async",
                    "nvvm.wgmma.commit.group.sync.aligned",
                    "nvvm.wgmma.wait.group.sync.aligned",
                ],
                2 => vec![
                    "nvvm.fence.proxy",
                    "nvvm.wgmma.fence.aligned",
                    "nvvm.wgmma.mma_async",
                    "nvvm.wgmma.commit.group.sync.aligned",
                    "nvvm.wgmma.mma_async",
                    "nvvm.wgmma.commit.group.sync.aligned",
                    "nvvm.wgmma.wait.group.sync.aligned",
                    "nvvm.wgmma.wait.group.sync.aligned",
                ],
                8 => vec![
                    "nvvm.fence.proxy",
                    "nvvm.wgmma.fence.aligned",
                    "nvvm.wgmma.mma_async",
                    "nvvm.wgmma.commit.group.sync.aligned",
                    "nvvm.wgmma.mma_async",
                    "nvvm.wgmma.commit.group.sync.aligned",
                    "nvvm.wgmma.wait.group.sync.aligned",
                ],
                _ => unreachable!(),
            };
            assert_eq!(operations, expected);
            assert_eq!(shapes, vec!["#nvvm.shape<m = 64, n = 8, k = 16>", "#nvvm.shape<m = 64, n = 8, k = 16>"]);
            assert_eq!(layouts, vec![("#nvvm.mma_layout<row>".to_owned(), "#nvvm.mma_layout<col>".to_owned()); 2]);
            assert_eq!(
                groups,
                match stages {
                    1 => vec!["0 : i64", "0 : i64"],
                    2 => vec!["1 : i64", "0 : i64"],
                    8 => vec!["0 : i64"],
                    _ => unreachable!(),
                }
            );
        }
    }
    #[test]
    fn test_lowering_wgmma_scaled_operands() {
        let types = [
            ArrayType::new_static(DataType::BF16, [64, 32]),
            ArrayType::new_static(DataType::BF16, [32, 8]),
            ArrayType::new_static(DataType::BF16, [64, 2]),
            ArrayType::new_static(DataType::BF16, [4, 8]),
            ArrayType::new_static(DataType::F32, [64, 8]),
        ];
        let parameters = types
            .into_iter()
            .enumerate()
            .map(|(index, r#type)| {
                whole_array_parameter(
                    r#type,
                    if index == 4 { KernelParameterAccess::WriteOnly } else { KernelParameterAccess::ReadOnly },
                )
                .unwrap()
            })
            .collect();
        let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), parameters).unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(call, |(references, _)| {
            let inputs = references[..4].iter().map(|reference| reference.read()).collect::<Result<Vec<_>, _>>()?;
            let result = inputs[0]
                .context()
                .bind(
                    ArrayIrOperation::Array(ArrayOperation::ScaledDot(ScaledDotOperation::new(
                        DotOperation::matmul().dimensions().clone(),
                        DataType::F32,
                        true,
                        true,
                    ))),
                    vec![],
                    &inputs,
                )?
                .remove(0);
            references[4].write(&result)
        })
        .unwrap();
        let context = Context::new();
        let target = Target::new(9, 0).unwrap().with_threads_per_block(128).unwrap();
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let module = Compiler
            .module(&context, &kernel, &target, &Options::default().with_mma(Mma::Wgmma), &KernelSchedule::default())
            .unwrap();
        assert!(module.verify().unwrap());
        let mut operations = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            let name = operation.name().to_string();
            if matches!(name.as_str(), "arith.extf" | "arith.mulf" | "arith.truncf" | "nvvm.wgmma.mma_async") {
                operations.push((name, operation.result(0).unwrap().r#type().unwrap().to_string()));
            }
            WalkResult::Advance
        });
        assert_eq!(
            operations,
            vec![
                ("arith.extf".to_owned(), "f32".to_owned()),
                ("arith.extf".to_owned(), "f32".to_owned()),
                ("arith.mulf".to_owned(), "f32".to_owned()),
                ("arith.truncf".to_owned(), "bf16".to_owned()),
                ("arith.extf".to_owned(), "f32".to_owned()),
                ("arith.extf".to_owned(), "f32".to_owned()),
                ("arith.mulf".to_owned(), "f32".to_owned()),
                ("arith.truncf".to_owned(), "bf16".to_owned()),
                ("nvvm.wgmma.mma_async".to_owned(), "!llvm.struct<(f32, f32, f32, f32)>".to_owned()),
                ("nvvm.wgmma.mma_async".to_owned(), "!llvm.struct<(f32, f32, f32, f32)>".to_owned()),
            ]
        );
        let bytes = crate::kernels::gpu::lowering::module::serialize(&module).unwrap();
        assert_eq!(&bytes[..4], b"ML\xefR");
    }
}
