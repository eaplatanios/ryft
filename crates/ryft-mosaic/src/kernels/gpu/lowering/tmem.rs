//! Datacenter Blackwell tensor-memory transport through native NVVM instructions.
//!
//! Instruction and shared descriptors follow PTX 9.2 sections 9.7.16.4.1–2. Operand packing uses eight-row atoms
//! with 32 bytes per row, the 32-byte swizzle, and 256-byte base alignment. All 32 lanes of warp zero allocate and
//! deallocate collectively; the elected CTA's thread zero issues MMA and commits its completion barriers. After
//! completion, each CTA's four warps load their own 128 accumulator rows using `tcgen05.ld.32x32b` and `wait::ld`.
//! Cooperative kernels then exchange row halves through cluster shared memory. Scale copies preserve the separate
//! replicated `32x128b.warpx4` layout and become readable only after their explicit completion wait.

use ryft_core::{ArrayReferenceView, ArraySliceAxis, ArrayType, DataType, ValueId};
use ryft_mlir::dialects::builtin::VectorTypeDimension;
use ryft_mlir::dialects::{arith, llvm, memref, nvvm, scf};
use ryft_mlir::{Attribute, Block, DetachedBlock, Type, TypeRef, UnknownLocationRef};

use crate::kernels::gpu::lowering::{Buffer, KernelValue, Lowering, append, shape};
use crate::kernels::gpu::synchronization::SynchronizationEvent;
use crate::kernels::gpu::{Error, Target};

/// A native tensor-memory allocation, kept separate from ordinary memrefs and their address calculations.
#[derive(Clone)]
pub(super) struct Tmem<'c, 't> {
    /// Canonical allocation result used by lifetime and synchronization analysis.
    pub(super) owner: ValueId,

    /// Native address-space-six pointer returned by tensor-memory allocation.
    pub(super) address: KernelValue<'c, 't>,

    /// Allocated columns; every column holds 128 FP32 elements.
    pub(super) columns: usize,

    /// Signed scale bits owned by this same logical allocation; native scale storage holds their magnitudes.
    pub(super) scale_signs: Option<Buffer<'c, 't>>,
}

/// One pending multiplication and its shared completion barrier.
#[derive(Clone)]
pub(super) struct TmemToken<'c, 't> {
    /// Canonical completion-token result.
    pub(super) owner: ValueId,

    /// Shared mbarrier initialized for one completion arrival.
    pub(super) barrier: KernelValue<'c, 't>,

    /// Whether the issuing thread has attached the multiplication to this barrier.
    pub(super) committed: bool,
}

/// Checks the architecture and full participating CTA required by the initial pinned NVVM instruction set.
pub(super) fn admit(target: &Target) -> Result<(), Error> {
    if !matches!(target.compute_capability(), (10, 0) | (10, 1) | (11, 0)) {
        return Err(unsupported("tcgen05 requires compute capability 10.0, 10.1, or 11.0 in the pinned compiler"));
    }
    if target.threads_per_block() != 128 {
        return Err(unsupported("tcgen05 requires exactly 128 threads per block"));
    }
    Ok(())
}

/// Returns the exact two K-fastest shared transports for already type-checked MMA operands.
pub(super) fn requirements(inputs: &[ArrayType]) -> Result<[ArrayType; 2], Error> {
    if inputs.len() != 2 {
        return Err(unsupported("tcgen05 requires two multiplicand arrays"));
    }
    let left = shape(&inputs[0])?;
    let right = shape(&inputs[1])?;
    if left.len() != 2 || right.len() != 2 || !matches!(left[0], 128 | 256) || left[1] >= 16_384 {
        return Err(unsupported("tcgen05 shared-memory stride exceeds the descriptor encoding"));
    }
    Ok([
        ArrayType::new_static(inputs[0].data_type(), [128 * left[1]]),
        ArrayType::new_static(inputs[1].data_type(), [right[0] * right[1] / (left[0] / 128)]),
    ])
}

/// Returns the native column quota for the replicated scale layout, rounded to a legal tensor-memory allocation.
pub(super) fn scale_columns(rows: usize, blocks: usize) -> Result<usize, Error> {
    if !(32..=256).contains(&rows) || rows % 32 != 0 || blocks == 0 || blocks > 128 {
        return Err(unsupported("scale storage requires rows divisible by 32 in [32, 256] and blocks in [1, 128]"));
    }
    Ok((rows.div_ceil(128) * blocks.div_ceil(4) * 4).max(32).next_power_of_two())
}

/// Returns the flat byte transport for row-padded, four-scale groups in native `32x128b.warpx4` copy order.
pub(super) fn scale_requirements(source: &ArrayType) -> Result<ArrayType, Error> {
    let dimensions = shape(source)?;
    if !matches!(source.data_type(), DataType::F8E8M0FNU | DataType::F8E4M3FN) || dimensions.len() != 2 {
        return Err(unsupported("scale transport requires a rank-two F8E8M0FNU or F8E4M3FN array"));
    }
    scale_columns(dimensions[0], dimensions[1])?;
    Ok(ArrayType::new_static(DataType::U8, [dimensions[0].div_ceil(128) * dimensions[1].div_ceil(4) * 512]))
}

/// Constructs an exact adapter-owned admission or protocol error.
fn unsupported(reason: &str) -> Error {
    Error::Unsupported { operation: "mosaic_gpu.tmem", reason: reason.to_owned() }
}

impl<'c, 't> Lowering<'c, 't> {
    /// Allocates columns collectively in warp zero and broadcasts the returned address through its shared slot.
    pub(super) fn tmem_allocate(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        owner: ValueId,
        columns: usize,
        address_slot: KernelValue<'c, 't>,
        scale_signs: Option<Buffer<'c, 't>>,
    ) -> Result<Tmem<'c, 't>, Error> {
        for thread in 0..32 {
            self.tmem_event(thread, SynchronizationEvent::AllocateTensorMemory { value: owner })?;
        }
        let pointer = self.tmem_shared_pointer(block, address_slot)?;
        self.tmem_participants(block, true, false, |lowering, body| {
            let columns = lowering.tmem_integer(body, 32, columns as u64)?;
            body.append_operation(nvvm::tcgen05_alloc(
                &[pointer, columns],
                &[],
                &[("group", lowering.context.nvvm_cta_group_kind_attribute(lowering.tmem_group())?.as_ref())],
                false,
                lowering.location,
            )?)?;
            Ok(())
        })?;
        self.barrier(block)?;
        let zero = self.index(block, 0)?;
        let address = append(
            block,
            memref::load(address_slot, &[zero], self.context.signless_integer_type(32), false, None, self.location)?,
        )?;
        let address = append(block, llvm::inttoptr(address, self.context.llvm_pointer_type(6)?, self.location)?)?;
        Ok(Tmem { owner, address, columns, scale_signs })
    }

    /// Packs immutable operands, initializes the completion barrier, and issues native asynchronous matrix products.
    pub(super) fn tmem_mma(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        inputs: &[Buffer<'c, 't>; 2],
        destination: &Tmem<'c, 't>,
        owner: ValueId,
        barrier_slot: KernelValue<'c, 't>,
        transport: &[KernelValue<'c, 't>; 2],
        accumulate: bool,
        scales: Option<&[Tmem<'c, 't>; 2]>,
    ) -> Result<TmemToken<'c, 't>, Error> {
        let packed = inputs[0].r#type.data_type() == DataType::U8;
        if packed && scales.is_none_or(|scales| scales.iter().any(|scale| scale.scale_signs.is_none())) {
            return Err(unsupported("nvfp4 multiplication requires allocation-owned signed scale storage"));
        }
        let element_bytes = if scales.is_some() { 1 } else { 2 };
        let contraction = shape(&inputs[0].r#type)?[1];
        let columns = destination.columns;
        let collective = self.cluster_rank.is_some();
        for (operand, source) in inputs.iter().enumerate() {
            let contraction_minor = operand == 0 || packed;
            let local_major = if operand == 0 { 128 } else { columns / if collective { 2 } else { 1 } };
            let source_columns = if contraction_minor { contraction } else { columns };
            let local_columns = if contraction_minor { contraction } else { local_major };
            self.distributed(block, local_major * contraction, |lowering, body, index| {
                let width = lowering.index(body, local_columns)?;
                let row = append(body, arith::divui(index, width, lowering.location)?)?;
                let column = append(body, arith::remui(index, width, lowering.location)?)?;
                let (major, reduction) = if contraction_minor { (row, column) } else { (column, row) };
                let swizzled = lowering.mma_swizzled_index(body, major, reduction, contraction, element_bytes)?;
                let (source_row, source_column) = if let Some(rank) = lowering.cluster_rank {
                    let extent = lowering.index(body, local_major)?;
                    let offset = append(body, arith::muli(rank, extent, lowering.location)?)?;
                    if contraction_minor {
                        (append(body, arith::addi(row, offset, lowering.location)?)?, column)
                    } else {
                        (row, append(body, arith::addi(column, offset, lowering.location)?)?)
                    }
                } else {
                    (row, column)
                };
                let width = lowering.index(body, source_columns)?;
                let source_index = append(body, arith::muli(source_row, width, lowering.location)?)?;
                let source_index = append(body, arith::addi(source_index, source_column, lowering.location)?)?;
                let value = lowering.load(body, source, source_index)?;
                let value = if packed {
                    let signs = scales.unwrap()[operand].scale_signs.as_ref().unwrap();
                    let block_bytes = lowering.index(body, 8)?;
                    let scale_column = append(body, arith::divui(source_column, block_bytes, lowering.location)?)?;
                    let scale_columns = lowering.index(body, contraction / 8)?;
                    let scale_index = append(body, arith::muli(source_row, scale_columns, lowering.location)?)?;
                    let scale_index = append(body, arith::addi(scale_index, scale_column, lowering.location)?)?;
                    let sign = lowering.load(body, signs, scale_index)?;
                    let shift = lowering.tmem_integer(body, 8, 4)?;
                    let low_sign = append(body, arith::shrui(sign, shift, lowering.location)?)?;
                    let both_signs = append(body, arith::ori(sign, low_sign, lowering.location)?)?;
                    append(body, arith::xori(value, both_signs, lowering.location)?)?
                } else {
                    value
                };
                body.append_operation(memref::store(
                    value,
                    transport[operand],
                    &[swizzled],
                    false,
                    None,
                    lowering.location,
                )?)?;
                Ok(())
            })?;
        }
        let barrier = self.tmem_shared_pointer(block, barrier_slot)?;
        self.tmem_participants(block, false, false, |lowering, body| {
            let one = lowering.tmem_integer(body, 32, 1)?;
            body.append_operation(nvvm::mbarrier_init(&[barrier, one], &[], &[], false, lowering.location)?)?;
            body.append_operation(nvvm::fence_mbarrier_init(&[], &[], &[], false, lowering.location)?)?;
            Ok(())
        })?;
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
        self.barrier(block)?;
        block.append_operation(nvvm::tcgen05_fence(
            &[],
            &[],
            &[("kind", self.context.nvvm_tcgen05_fence_kind_attribute(nvvm::Tcgen05FenceKind::After)?.as_ref())],
            false,
            self.location,
        )?)?;
        let descriptors = [
            self.tmem_descriptor(block, transport[0], contraction * element_bytes)?,
            self.tmem_descriptor(block, transport[1], contraction * element_bytes)?,
        ];
        let format = u64::from(inputs[0].r#type.data_type() == DataType::BF16);
        let descriptor = if scales.is_some() {
            (if packed { (1 << 7) | (1 << 10) } else { 1 << 23 })
                | ((columns as u64 / 8) << 17)
                | (if collective { 2 } else { 1 } << 27)
        } else {
            (1 << 4)
                | (format << 7)
                | (format << 10)
                | ((columns as u64 / 8) << 17)
                | (if collective { 16 } else { 8 } << 24)
        };
        self.tmem_participants(block, false, true, |lowering, body| {
            for group in 0..contraction / (32 / element_bytes) {
                let descriptor = lowering.tmem_integer(
                    body,
                    32,
                    descriptor
                        | if scales.is_some() && !packed {
                            ((group as u64 % 4) << 4) | ((group as u64 % 4) << 29)
                        } else {
                            0
                        },
                )?;
                let offset = lowering.tmem_integer(body, 64, (group * 16) as u64)?;
                let left = append(body, arith::addi(descriptors[0], offset, lowering.location)?)?;
                let right = append(body, arith::addi(descriptors[1], offset, lowering.location)?)?;
                let enabled = lowering.tmem_integer(body, 1, u64::from(accumulate || group != 0))?;
                if let Some(scales) = scales {
                    let scale_group = if packed { group } else { group / 4 };
                    let left_scale = lowering.tmem_offset(
                        body,
                        scales[0].address,
                        scale_group * 4 * if collective { 2 } else { 1 },
                    )?;
                    let right_scale =
                        lowering.tmem_offset(body, scales[1].address, scale_group * 4 * columns.div_ceil(128))?;
                    body.append_operation(nvvm::tcgen05_mma_block_scale(
                        &[destination.address, left, right, descriptor, enabled, left_scale, right_scale],
                        &[],
                        &[
                            (
                                "blockScale",
                                lowering
                                    .context
                                    .nvvm_tcgen05_mma_block_scale_attribute(if packed {
                                        nvvm::Tcgen05MmaBlockScale::Block16
                                    } else {
                                        nvvm::Tcgen05MmaBlockScale::Block32
                                    })?
                                    .as_ref(),
                            ),
                            (
                                "kind",
                                lowering
                                    .context
                                    .nvvm_tcgen05_mma_kind_attribute(if packed {
                                        nvvm::Tcgen05MmaKind::Mxf4nvf4
                                    } else {
                                        nvvm::Tcgen05MmaKind::Mxf8f6f4
                                    })?
                                    .as_ref(),
                            ),
                            (
                                "ctaGroup",
                                lowering.context.nvvm_cta_group_kind_attribute(lowering.tmem_group())?.as_ref(),
                            ),
                        ],
                        false,
                        lowering.location,
                    )?)?;
                } else {
                    body.append_operation(nvvm::tcgen05_mma(
                        &[destination.address, left, right, descriptor, enabled],
                        &[],
                        &[
                            (
                                "kind",
                                lowering.context.nvvm_tcgen05_mma_kind_attribute(nvvm::Tcgen05MmaKind::F16)?.as_ref(),
                            ),
                            (
                                "ctaGroup",
                                lowering.context.nvvm_cta_group_kind_attribute(lowering.tmem_group())?.as_ref(),
                            ),
                            (
                                "operandSegmentSizes",
                                lowering.context.dense_i32_array_attribute(&[1, 1, 1, 1, 1, 0, 0])?.as_ref(),
                            ),
                        ],
                        false,
                        lowering.location,
                    )?)?;
                }
            }
            Ok(())
        })?;
        if collective {
            let left = [0, 1].map(|rank| {
                (
                    inputs[0].owner,
                    ArrayReferenceView::Slice {
                        axes: vec![ArraySliceAxis::new(rank * 128, 128, 1), ArraySliceAxis::new(0, contraction, 1)],
                    },
                )
            });
            let right = [0, 1].map(|rank| {
                (
                    inputs[1].owner,
                    ArrayReferenceView::Slice {
                        axes: vec![
                            ArraySliceAxis::new(
                                if packed { rank * columns / 2 } else { 0 },
                                if packed { columns / 2 } else { contraction },
                                1,
                            ),
                            ArraySliceAxis::new(
                                if packed { 0 } else { rank * columns / 2 },
                                if packed { contraction } else { columns / 2 },
                                1,
                            ),
                        ],
                    },
                )
            });
            self.record_synchronization(
                Some(0),
                0,
                SynchronizationEvent::IssueTensorMemoryCluster {
                    token: owner,
                    destination: destination.owner,
                    left,
                    right,
                    accumulate,
                    scales: scales.map(|scales| scales.iter().map(|scale| scale.owner).collect()).unwrap_or_default(),
                },
            )?;
        } else {
            self.tmem_event(
                0,
                SynchronizationEvent::IssueTensorMemory {
                    token: owner,
                    destination: destination.owner,
                    left: (
                        inputs[0].owner,
                        ArrayReferenceView::Slice {
                            axes: shape(&inputs[0].r#type)?
                                .into_iter()
                                .map(|size| ArraySliceAxis::new(0, size, 1))
                                .collect(),
                        },
                    ),
                    right: (
                        inputs[1].owner,
                        ArrayReferenceView::Slice {
                            axes: shape(&inputs[1].r#type)?
                                .into_iter()
                                .map(|size| ArraySliceAxis::new(0, size, 1))
                                .collect(),
                        },
                    ),
                    accumulate,
                    scales: scales.map(|scales| scales.iter().map(|scale| scale.owner).collect()).unwrap_or_default(),
                },
            )?;
        }
        Ok(TmemToken { owner, barrier, committed: false })
    }

    /// Packs logical scale rows into four-scale groups, pads absent lanes, and copies them into the replicated scale
    /// layout with native `tcgen05.cp.32x128b.warpx4`. Completion remains pending until explicit commit and wait.
    pub(super) fn tmem_scale_copy(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        source: &Buffer<'c, 't>,
        destination: &Tmem<'c, 't>,
        owner: ValueId,
        barrier_slot: KernelValue<'c, 't>,
        transport: KernelValue<'c, 't>,
    ) -> Result<TmemToken<'c, 't>, Error> {
        let dimensions = shape(&source.r#type)?;
        let rows = dimensions[0];
        let blocks = dimensions[1];
        let row_tiles = rows.div_ceil(128);
        let block_tiles = blocks.div_ceil(4);
        self.distributed(block, row_tiles * block_tiles * 512, |lowering, body, index| {
            let tile_bytes = lowering.index(body, 512)?;
            let sixteen = lowering.index(body, 16)?;
            let four = lowering.index(body, 4)?;
            let thirty_two = lowering.index(body, 32)?;
            let one_twenty_eight = lowering.index(body, 128)?;
            let block_tiles = lowering.index(body, block_tiles)?;
            let tile = append(body, arith::divui(index, tile_bytes, lowering.location)?)?;
            let within = append(body, arith::remui(index, tile_bytes, lowering.location)?)?;
            let lane = append(body, arith::divui(within, sixteen, lowering.location)?)?;
            let packed_column = append(body, arith::remui(within, sixteen, lowering.location)?)?;
            let quarter = append(body, arith::divui(packed_column, four, lowering.location)?)?;
            let block_lane = append(body, arith::remui(packed_column, four, lowering.location)?)?;
            let row_tile = append(body, arith::divui(tile, block_tiles, lowering.location)?)?;
            let block_tile = append(body, arith::remui(tile, block_tiles, lowering.location)?)?;
            let row_base = append(body, arith::muli(row_tile, one_twenty_eight, lowering.location)?)?;
            let quarter_base = append(body, arith::muli(quarter, thirty_two, lowering.location)?)?;
            let row = append(body, arith::addi(row_base, quarter_base, lowering.location)?)?;
            let row = append(body, arith::addi(row, lane, lowering.location)?)?;
            let column_base = append(body, arith::muli(block_tile, four, lowering.location)?)?;
            let column = append(body, arith::addi(column_base, block_lane, lowering.location)?)?;
            let row_bound = lowering.index(body, rows)?;
            let column_bound = lowering.index(body, blocks)?;
            let valid_row = append(
                body,
                arith::cmpi(row, row_bound, arith::IntegerComparisonPredicate::UnsignedLessThan, lowering.location)?,
            )?;
            let valid_column = append(
                body,
                arith::cmpi(
                    column,
                    column_bound,
                    arith::IntegerComparisonPredicate::UnsignedLessThan,
                    lowering.location,
                )?,
            )?;
            let valid = append(body, arith::andi(valid_row, valid_column, lowering.location)?)?;
            let integer = lowering.context.signless_integer_type(8);
            let mut present = lowering.context.block::<TypeRef<'c, 't>, UnknownLocationRef<'c, 't>>(&[]);
            let row_start = append(&mut present, arith::muli(row, column_bound, lowering.location)?)?;
            let source_index = append(&mut present, arith::addi(row_start, column, lowering.location)?)?;
            let value = lowering.load(&mut present, source, source_index)?;
            let value = if let Some(signs) = &destination.scale_signs {
                let sign_mask = lowering.tmem_integer(&mut present, 8, 0x80)?;
                let sign = append(&mut present, arith::andi(value, sign_mask, lowering.location)?)?;
                lowering.store(&mut present, signs, source_index, sign)?;
                let magnitude_mask = lowering.tmem_integer(&mut present, 8, 0x7f)?;
                append(&mut present, arith::andi(value, magnitude_mask, lowering.location)?)?
            } else {
                value
            };
            present.append_operation(scf::r#yield(&[value], lowering.location)?)?;
            let mut absent = lowering.context.block::<TypeRef<'c, 't>, UnknownLocationRef<'c, 't>>(&[]);
            let zero = lowering.tmem_integer(&mut absent, 8, 0)?;
            absent.append_operation(scf::r#yield(&[zero], lowering.location)?)?;
            let value = append(
                body,
                scf::r#if(
                    valid,
                    &[integer.as_ref()],
                    present.try_into()?,
                    Some(absent.try_into()?),
                    lowering.location,
                )?,
            )?;
            body.append_operation(memref::store(value, transport, &[index], false, None, lowering.location)?)?;
            Ok(())
        })?;
        let barrier = self.tmem_shared_pointer(block, barrier_slot)?;
        self.tmem_participants(block, false, false, |lowering, body| {
            let one = lowering.tmem_integer(body, 32, 1)?;
            body.append_operation(nvvm::mbarrier_init(&[barrier, one], &[], &[], false, lowering.location)?)?;
            body.append_operation(nvvm::fence_mbarrier_init(&[], &[], &[], false, lowering.location)?)?;
            Ok(())
        })?;
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
        self.barrier(block)?;
        let address = self.mma_descriptor_address(block, transport)?;
        let fields = self.tmem_integer(block, 64, (1 << 46) | (8 << 32))?;
        let descriptor = append(block, arith::ori(address, fields, self.location)?)?;
        self.tmem_participants(block, false, true, |lowering, body| {
            for row_tile in 0..row_tiles {
                for block_tile in 0..block_tiles {
                    let offset =
                        lowering.tmem_integer(body, 64, ((row_tile * block_tiles + block_tile) * 32) as u64)?;
                    let source = append(body, arith::addi(descriptor, offset, lowering.location)?)?;
                    let destination =
                        lowering.tmem_offset(body, destination.address, 4 * row_tiles * block_tile + 4 * row_tile)?;
                    body.append_operation(nvvm::tcgen05_cp(
                        &[destination, source],
                        &[],
                        &[
                            ("group", lowering.context.nvvm_cta_group_kind_attribute(lowering.tmem_group())?.as_ref()),
                            (
                                "shape",
                                lowering
                                    .context
                                    .nvvm_tcgen05_cp_shape_attribute(nvvm::Tcgen05CpShape::Shape32x128b)?
                                    .as_ref(),
                            ),
                            (
                                "multicast",
                                lowering
                                    .context
                                    .nvvm_tcgen05_cp_multicast_attribute(nvvm::Tcgen05CpMulticast::Warpx4)?
                                    .as_ref(),
                            ),
                        ],
                        false,
                        lowering.location,
                    )?)?;
                }
            }
            Ok(())
        })?;
        if self.cluster_rank.is_some() {
            self.record_synchronization(
                Some(0),
                0,
                SynchronizationEvent::CopyTensorMemoryCluster {
                    token: owner,
                    destination: destination.owner,
                    source: (
                        source.owner,
                        ArrayReferenceView::Slice {
                            axes: vec![ArraySliceAxis::new(0, rows, 1), ArraySliceAxis::new(0, blocks, 1)],
                        },
                    ),
                },
            )?;
        } else {
            self.tmem_event(
                0,
                SynchronizationEvent::CopyTensorMemory {
                    token: owner,
                    destination: destination.owner,
                    source: (
                        source.owner,
                        ArrayReferenceView::Slice {
                            axes: vec![ArraySliceAxis::new(0, rows, 1), ArraySliceAxis::new(0, blocks, 1)],
                        },
                    ),
                },
            )?;
        }
        Ok(TmemToken { owner, barrier, committed: false })
    }

    /// Attaches the issuing thread's pending multiplication to the token's completion barrier exactly once.
    pub(super) fn tmem_commit(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        token: &mut TmemToken<'c, 't>,
    ) -> Result<(), Error> {
        if token.committed {
            return Err(unsupported("tensor-memory completion token has already been committed"));
        }
        self.tmem_participants(block, false, true, |lowering, body| {
            let mut operands = vec![token.barrier];
            if lowering.cluster_rank.is_some() {
                operands.push(lowering.tmem_integer(body, 16, 3)?);
            }
            body.append_operation(nvvm::tcgen05_commit(
                &operands,
                &[],
                &[("group", lowering.context.nvvm_cta_group_kind_attribute(lowering.tmem_group())?.as_ref())],
                false,
                lowering.location,
            )?)?;
            Ok(())
        })?;
        if self.cluster_rank.is_some() {
            self.record_synchronization(
                Some(0),
                0,
                SynchronizationEvent::CommitTensorMemoryCluster { token: token.owner },
            )?;
        } else {
            self.tmem_event(0, SynchronizationEvent::CommitTensorMemory { token: token.owner })?;
        }
        token.committed = true;
        Ok(())
    }

    /// Waits for the committed multiplication before retiring its native barrier and publishing writes to the CTA.
    pub(super) fn tmem_wait(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        token: TmemToken<'c, 't>,
    ) -> Result<(), Error> {
        if !token.committed {
            return Err(unsupported("tensor-memory completion token must be committed before waiting"));
        }
        self.tmem_event(0, SynchronizationEvent::WaitTensorMemory { token: token.owner })?;
        self.tmem_participants(block, false, false, |lowering, body| {
            let phase = lowering.tmem_integer(body, 32, 0)?;
            let ticks = lowering.tmem_integer(body, 32, 10_000_000)?;
            body.append_operation(nvvm::mbarrier_try_wait_parity(
                &[token.barrier, phase, ticks],
                &[],
                &[],
                false,
                lowering.location,
            )?)?;
            body.append_operation(nvvm::tcgen05_fence(
                &[],
                &[],
                &[(
                    "kind",
                    lowering.context.nvvm_tcgen05_fence_kind_attribute(nvvm::Tcgen05FenceKind::Before)?.as_ref(),
                )],
                false,
                lowering.location,
            )?)?;
            Ok(())
        })?;
        self.barrier(block)?;
        block.append_operation(nvvm::tcgen05_fence(
            &[],
            &[],
            &[("kind", self.context.nvvm_tcgen05_fence_kind_attribute(nvvm::Tcgen05FenceKind::After)?.as_ref())],
            false,
            self.location,
        )?)?;
        self.tmem_participants(block, false, false, |lowering, body| {
            body.append_operation(nvvm::mbarrier_inval(&[token.barrier], &[], &[], false, lowering.location)?)?;
            Ok(())
        })?;
        self.barrier(block)
    }

    /// Loads one tensor-memory row per CTA thread, with explicit per-thread completion before register use.
    pub(super) fn tmem_load(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        source: &Tmem<'c, 't>,
        output: &Buffer<'c, 't>,
    ) -> Result<(), Error> {
        for rank in 0..if self.cluster_rank.is_some() { 2 } else { 1 } {
            for thread in 0..128 {
                self.record_synchronization(
                    Some(rank),
                    thread,
                    SynchronizationEvent::LoadTensorMemory {
                        value: source.owner,
                        view: ArrayReferenceView::Slice {
                            axes: vec![
                                ArraySliceAxis::new(rank * 128 + thread as usize, 1, 1),
                                ArraySliceAxis::new(0, source.columns, 1),
                            ],
                        },
                    },
                )?;
            }
        }
        let integer = self.context.signless_integer_type(32);
        let vector = self.context.vector_type(integer, &[VectorTypeDimension::Fixed(1)], self.location)?;
        let width = self.index(block, source.columns)?;
        let row = if let Some(rank) = self.cluster_rank {
            let rows = self.index(block, 128)?;
            let offset = append(block, arith::muli(rank, rows, self.location)?)?;
            append(block, arith::addi(self.thread, offset, self.location)?)?
        } else {
            self.thread
        };
        let row_start = append(block, arith::muli(row, width, self.location)?)?;
        let warp_size = self.index(block, 32)?;
        let warp = append(block, arith::divui(self.thread, warp_size, self.location)?)?;
        let warp = append(block, arith::index_castui(warp, integer, self.location)?)?;
        let row_stride = self.tmem_integer(block, 32, 32 << 16)?;
        let warp_offset = append(block, arith::muli(warp, row_stride, self.location)?)?;
        let base = append(block, llvm::ptrtoint(source.address, integer, self.location)?)?;
        let base = append(block, arith::addi(base, warp_offset, self.location)?)?;
        for column in 0..source.columns {
            let column_offset = self.tmem_integer(block, 32, column as u64)?;
            let address = append(block, arith::addi(base, column_offset, self.location)?)?;
            let address = append(block, llvm::inttoptr(address, self.context.llvm_pointer_type(6)?, self.location)?)?;
            let values = append(
                block,
                nvvm::tcgen05_ld(
                    &[address],
                    &[vector.as_ref()],
                    &[(
                        "shape",
                        self.context.nvvm_tcgen05_ld_st_shape_attribute(nvvm::Tcgen05LdStShape::Shape32x32b)?.as_ref(),
                    )],
                    false,
                    self.location,
                )?,
            )?;
            block.append_operation(nvvm::tcgen05_wait(
                &[],
                &[],
                &[("kind", self.context.nvvm_tcgen05_wait_kind_attribute(nvvm::Tcgen05WaitKind::Load)?.as_ref())],
                false,
                self.location,
            )?)?;
            let zero = self.tmem_integer(block, 32, 0)?;
            let bits = append(block, llvm::extract_element(values, zero, integer.as_ref(), self.location)?)?;
            let value = append(block, arith::bitcast(bits, self.context.float32_type(), self.location)?)?;
            let column = self.index(block, column)?;
            let index = append(block, arith::addi(row_start, column, self.location)?)?;
            self.store(block, output, index, value)?;
        }
        if self.cluster_rank.is_some() { self.cluster_gather_rows(block, output, 128) } else { self.barrier(block) }
    }

    /// Releases the allocation collectively after every thread has completed its tensor-memory loads.
    pub(super) fn tmem_release(&mut self, block: &mut DetachedBlock<'c, 't>, value: Tmem<'c, 't>) -> Result<(), Error> {
        self.barrier(block)?;
        for thread in 0..32 {
            self.tmem_event(thread, SynchronizationEvent::ReleaseTensorMemory { value: value.owner })?;
        }
        self.tmem_participants(block, true, false, |lowering, body| {
            let columns = lowering.tmem_integer(body, 32, value.columns as u64)?;
            body.append_operation(nvvm::tcgen05_dealloc(
                &[value.address, columns],
                &[],
                &[("group", lowering.context.nvvm_cta_group_kind_attribute(lowering.tmem_group())?.as_ref())],
                false,
                lowering.location,
            )?)?;
            Ok(())
        })?;
        self.barrier(block)
    }

    /// Relinquishes the CTA's allocation permit after its final logical grid iteration and all releases.
    pub(super) fn tmem_relinquish(&mut self, block: &mut DetachedBlock<'c, 't>) -> Result<(), Error> {
        self.tmem_participants(block, true, false, |lowering, body| {
            body.append_operation(nvvm::tcgen05_relinquish_alloc_permit(
                &[],
                &[],
                &[("group", lowering.context.nvvm_cta_group_kind_attribute(lowering.tmem_group())?.as_ref())],
                false,
                lowering.location,
            )?)?;
            Ok(())
        })?;
        self.barrier(block)
    }

    /// Selects the single native CTA group used by every tensor-memory instruction in this launch.
    fn tmem_group(&self) -> nvvm::CtaGroupKind {
        if self.cluster_rank.is_some() { nvvm::CtaGroupKind::Cta2 } else { nvvm::CtaGroupKind::Cta1 }
    }

    /// Records the actual participants and canonical ownership of a tensor-memory instruction.
    fn tmem_event(&mut self, thread: u32, event: SynchronizationEvent) -> Result<(), Error> {
        self.record_synchronization(None, thread, event)
    }

    /// Encodes aligned shared operands in K-fastest atoms with the required fixed descriptor bit and 32-byte swizzle.
    fn tmem_descriptor(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        value: KernelValue<'c, 't>,
        row_bytes: usize,
    ) -> Result<KernelValue<'c, 't>, Error> {
        let address = self.mma_descriptor_address(block, value)?;
        let fields = self.tmem_integer(block, 64, ((row_bytes as u64 / 2) << 32) | (1 << 46) | (6 << 61))?;
        append(block, arith::ori(address, fields, self.location)?)
    }

    /// Adds a native tensor-memory column offset without applying ordinary pointer element arithmetic.
    fn tmem_offset(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        address: KernelValue<'c, 't>,
        columns: usize,
    ) -> Result<KernelValue<'c, 't>, Error> {
        let integer = self.context.signless_integer_type(32);
        let address = append(block, llvm::ptrtoint(address, integer, self.location)?)?;
        let columns = self.tmem_integer(block, 32, columns as u64)?;
        let address = append(block, arith::addi(address, columns, self.location)?)?;
        append(block, llvm::inttoptr(address, self.context.llvm_pointer_type(6)?, self.location)?)
    }

    /// Converts a shared memref into the explicit pointer type required by allocation and barrier instructions.
    fn tmem_shared_pointer(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        value: KernelValue<'c, 't>,
    ) -> Result<KernelValue<'c, 't>, Error> {
        let address = append(block, memref::extract_aligned_pointer_as_index(value, self.location)?)?;
        let address =
            append(block, arith::index_castui(address, self.context.signless_integer_type(64), self.location)?)?;
        append(block, llvm::inttoptr(address, self.context.llvm_pointer_type(3)?, self.location)?)
    }

    /// Constructs exact instruction descriptor and control operands at their native integer widths.
    fn tmem_integer(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        width: usize,
        value: u64,
    ) -> Result<KernelValue<'c, 't>, Error> {
        append(
            block,
            arith::constant(
                self.context.integer_attribute(self.context.signless_integer_type(width), value as i64),
                self.location,
            )?,
        )
    }

    /// Selects the full first warp for collective resource operations, or its first thread for async issue/control.
    fn tmem_participants<F>(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        warp: bool,
        leader: bool,
        function: F,
    ) -> Result<(), Error>
    where
        F: FnOnce(&mut Self, &mut DetachedBlock<'c, 't>) -> Result<(), Error>,
    {
        let limit = self.index(block, if warp { 32 } else { 1 })?;
        let predicate = append(
            block,
            arith::cmpi(self.thread, limit, arith::IntegerComparisonPredicate::UnsignedLessThan, self.location)?,
        )?;
        let predicate = if let Some(rank) = self.cluster_rank.filter(|_| leader) {
            let zero = self.index(block, 0)?;
            let elected =
                append(block, arith::cmpi(rank, zero, arith::IntegerComparisonPredicate::Equal, self.location)?)?;
            append(block, arith::andi(predicate, elected, self.location)?)?
        } else {
            predicate
        };
        let mut body = self.context.block::<TypeRef<'c, 't>, UnknownLocationRef<'c, 't>>(&[]);
        function(self, &mut body)?;
        body.append_operation(scf::r#yield(&[], self.location)?)?;
        block.append_operation(scf::r#if(predicate, &[], body.try_into()?, None, self.location)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_admit() {
        for (major, minor) in [(10, 0), (10, 1), (11, 0)] {
            admit(&Target::new(major, minor).unwrap().with_threads_per_block(128).unwrap()).unwrap();
        }
        for (major, minor) in [(9, 0), (10, 3), (12, 0), (12, 1)] {
            assert!(matches!(admit(&Target::new(major, minor).unwrap().with_threads_per_block(128).unwrap()),
                Err(Error::Unsupported { operation: "mosaic_gpu.tmem", reason })
                if reason == "tcgen05 requires compute capability 10.0, 10.1, or 11.0 in the pinned compiler"));
        }
        assert!(matches!(admit(&Target::new(10, 0).unwrap()),
            Err(Error::Unsupported { operation: "mosaic_gpu.tmem", reason })
            if reason == "tcgen05 requires exactly 128 threads per block"));
    }

    #[test]
    fn test_requirements() {
        let inputs = [ArrayType::new_static(DataType::F16, [128, 16]), ArrayType::new_static(DataType::F16, [16, 32])];
        assert_eq!(
            requirements(&inputs).unwrap(),
            [ArrayType::new_static(DataType::F16, [2048]), ArrayType::new_static(DataType::F16, [512])]
        );
        let collective =
            [ArrayType::new_static(DataType::F16, [256, 16]), ArrayType::new_static(DataType::F16, [16, 32])];
        assert_eq!(
            requirements(&collective).unwrap(),
            [ArrayType::new_static(DataType::F16, [2048]), ArrayType::new_static(DataType::F16, [256])]
        );
        assert!(matches!(requirements(&inputs[..1]),
            Err(Error::Unsupported { operation: "mosaic_gpu.tmem", reason }) if reason == "tcgen05 requires two multiplicand arrays"));
    }

    #[test]
    fn test_tmem_native_lifetime() {
        use ryft_core::kernels::{
            Grid, KernelCallOperation, KernelDefinition, KernelOperation, KernelParameterAccess, KernelSchedule,
            VerifiedKernel, whole_array_parameter,
        };
        use ryft_core::{Context as CoreContext, ReferenceRead, ReferenceWrite};
        use ryft_mlir::{Context, Operation, WalkOrder, WalkResult};

        use crate::kernels::gpu::{Compiler, GpuOperation, Options, TmemOperation};

        for (data_type, rows) in
            [(DataType::F16, 128), (DataType::BF16, 128), (DataType::F16, 256), (DataType::BF16, 256)]
        {
            let call = KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                vec![
                    whole_array_parameter(
                        ArrayType::new_static(data_type, [rows, 16]),
                        KernelParameterAccess::ReadOnly,
                    )
                    .unwrap(),
                    whole_array_parameter(ArrayType::new_static(data_type, [16, 32]), KernelParameterAccess::ReadOnly)
                        .unwrap(),
                    whole_array_parameter(
                        ArrayType::new_static(DataType::F32, [rows, 32]),
                        KernelParameterAccess::WriteOnly,
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
            let definition = KernelDefinition::<GpuOperation>::trace(call, |(references, _)| {
                let context = references[0].context();
                let left = references[0].read()?;
                let right = references[1].read()?;
                let destination = context
                    .bind(
                        KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Allocate {
                            rows: rows as u16,
                            columns: 32,
                        })),
                        vec![],
                        &[],
                    )?
                    .remove(0);
                let token = context
                    .bind(
                        KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Mma { accumulate: false })),
                        vec![],
                        &[left, right, destination.clone()],
                    )?
                    .remove(0);
                context.bind(
                    KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Commit)),
                    vec![],
                    &[token.clone()],
                )?;
                context.bind(KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Wait)), vec![], &[token])?;
                let result = context
                    .bind(
                        KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Load)),
                        vec![],
                        &[destination.clone()],
                    )?
                    .remove(0);
                references[2].write(&result)?;
                context.bind(
                    KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Release)),
                    vec![],
                    &[destination],
                )?;
                Ok(())
            })
            .unwrap();
            let verified = VerifiedKernel::new(&definition, 1).unwrap();
            let context = Context::new();
            let target = Target::new(10, 0)
                .unwrap()
                .with_threads_per_block(128)
                .unwrap()
                .with_blocks_per_cluster((rows / 128) as u32)
                .unwrap();
            let module = Compiler
                .module(&context, &verified, &target, &Options::default(), &KernelSchedule::default())
                .unwrap();
            assert_eq!(module.verify(), Ok(true));
            let mut operations = Vec::new();
            module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
                let name = operation.name().as_str().unwrap().to_owned();
                if name.starts_with("nvvm.tcgen05") || name == "nvvm.mbarrier.try_wait.parity" {
                    operations.push(name);
                }
                WalkResult::Advance
            });
            assert_eq!(
                &operations[..5],
                &[
                    "nvvm.tcgen05.alloc",
                    "nvvm.tcgen05.fence",
                    "nvvm.tcgen05.mma",
                    "nvvm.tcgen05.commit",
                    "nvvm.mbarrier.try_wait.parity"
                ]
            );
            assert_eq!(operations.iter().filter(|name| name.as_str() == "nvvm.tcgen05.ld").count(), 32);
            assert_eq!(operations.iter().filter(|name| name.as_str() == "nvvm.tcgen05.wait").count(), 32);
            assert_eq!(
                &operations[operations.len() - 2..],
                &["nvvm.tcgen05.dealloc", "nvvm.tcgen05.relinquish_alloc_permit"]
            );
        }
    }

    #[test]
    fn test_scale_columns() {
        assert_eq!(scale_columns(128, 1).unwrap(), 32);
        assert_eq!(scale_columns(256, 17).unwrap(), 64);
        assert_eq!(scale_columns(256, 128).unwrap(), 256);
        assert!(matches!(scale_columns(31, 4), Err(Error::Unsupported { operation: "mosaic_gpu.tmem", reason })
            if reason == "scale storage requires rows divisible by 32 in [32, 256] and blocks in [1, 128]"));
    }

    #[test]
    fn test_scale_requirements() {
        assert_eq!(
            scale_requirements(&ArrayType::new_static(DataType::F8E8M0FNU, [32, 1])).unwrap(),
            ArrayType::new_static(DataType::U8, [512])
        );
        assert_eq!(
            scale_requirements(&ArrayType::new_static(DataType::F8E8M0FNU, [256, 5])).unwrap(),
            ArrayType::new_static(DataType::U8, [2048])
        );
    }

    #[test]
    fn test_tmem_native_block_scaled_lifetime() {
        use ryft_core::kernels::{
            Grid, KernelCallOperation, KernelDefinition, KernelOperation, KernelParameterAccess, KernelSchedule,
            VerifiedKernel, whole_array_parameter,
        };
        use ryft_core::{
            AddOperation, Array, ArrayIrValue, ArrayOperation, Context as CoreContext, MulOperation, ReferenceRead,
            ReferenceWrite,
        };
        use ryft_mlir::{Context, Operation, WalkOrder, WalkResult};

        use crate::kernels::gpu::{Compiler, GpuOperation, Options, TmemOperation};

        for (rows, packed, post_scale) in
            [(128, false, false), (256, false, false), (128, true, false), (256, true, false), (128, true, true)]
        {
            let contraction = if packed { 128 } else { 32 };
            let blocks = contraction / if packed { 16 } else { 32 };
            let scale_type = if packed { DataType::F8E4M3FN } else { DataType::F8E8M0FNU };
            let types = [
                ArrayType::new_static(
                    if packed { DataType::U8 } else { DataType::F8E4M3FN },
                    [rows, contraction / if packed { 2 } else { 1 }],
                ),
                if packed {
                    ArrayType::new_static(DataType::U8, [32, contraction / 2])
                } else {
                    ArrayType::new_static(DataType::F8E4M3FN, [contraction, 32])
                },
                ArrayType::new_static(scale_type, [rows, blocks]),
                ArrayType::new_static(scale_type, [32, blocks]),
                ArrayType::new_static(DataType::F32, [rows, 32]),
            ];
            let call = KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                types
                    .into_iter()
                    .enumerate()
                    .map(|(index, r#type)| {
                        whole_array_parameter(
                            r#type,
                            if index == 4 { KernelParameterAccess::WriteOnly } else { KernelParameterAccess::ReadOnly },
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap();
            let definition = KernelDefinition::<GpuOperation>::trace(call, |(references, _)| {
                let context = references[0].context();
                let left = references[0].read()?;
                let right = references[1].read()?;
                let mut scales = Vec::new();
                for (index, rows) in [(2, rows as u16), (3, 32)] {
                    let source = references[index].read()?;
                    let destination = context
                        .bind(
                            KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::AllocateScales {
                                rows,
                                blocks: blocks as u16,
                                data_type: scale_type,
                            })),
                            vec![],
                            &[],
                        )?
                        .remove(0);
                    let token = context
                        .bind(
                            KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::CopyScales)),
                            vec![],
                            &[source, destination.clone()],
                        )?
                        .remove(0);
                    context.bind(
                        KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Commit)),
                        vec![],
                        &[token.clone()],
                    )?;
                    context.bind(
                        KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Wait)),
                        vec![],
                        &[token],
                    )?;
                    scales.push(destination);
                }
                let destination = context
                    .bind(
                        KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Allocate {
                            rows: rows as u16,
                            columns: 32,
                        })),
                        vec![],
                        &[],
                    )?
                    .remove(0);
                let token = context
                    .bind(
                        KernelOperation::Extension(GpuOperation::Tmem(if packed {
                            TmemOperation::MmaNvfp4 { accumulate: false }
                        } else {
                            TmemOperation::MmaBlockScaled { accumulate: false }
                        })),
                        vec![],
                        &[left, right, scales[0].clone(), scales[1].clone(), destination.clone()],
                    )?
                    .remove(0);
                context.bind(
                    KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Commit)),
                    vec![],
                    &[token.clone()],
                )?;
                context.bind(KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Wait)), vec![], &[token])?;
                let result = context
                    .bind(
                        KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Load)),
                        vec![],
                        &[destination.clone()],
                    )?
                    .remove(0);
                let result = if post_scale {
                    let scale = context.lift(ArrayIrValue::Array(Array::scalar(2.0f32)?))?;
                    let scaled = context
                        .bind(
                            KernelOperation::Portable(ArrayOperation::Mul(MulOperation::new()).into()),
                            vec![],
                            &[result, scale],
                        )?
                        .remove(0);
                    let addend = context.lift(ArrayIrValue::Array(Array::scalar(3.0f32)?))?;
                    context
                        .bind(
                            KernelOperation::Portable(ArrayOperation::Add(AddOperation::new()).into()),
                            vec![],
                            &[scaled, addend],
                        )?
                        .remove(0)
                } else {
                    result
                };
                references[4].write(&result)?;
                for reference in scales.into_iter().chain([destination]) {
                    context.bind(
                        KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Release)),
                        vec![],
                        &[reference],
                    )?;
                }
                Ok(())
            })
            .unwrap();
            let verified = VerifiedKernel::new(&definition, 1).unwrap();
            let context = Context::new();
            let target = Target::new(10, 0)
                .unwrap()
                .with_threads_per_block(128)
                .unwrap()
                .with_blocks_per_cluster((rows / 128) as u32)
                .unwrap();
            let target = if packed {
                target.with_maximum_shared_memory_bytes(131_072).unwrap()
            } else if rows == 256 {
                assert!(matches!(
                    Compiler.module(&context, &verified, &target, &Options::default(), &KernelSchedule::default()),
                    Err(Error::SharedMemory { required: 49_472, maximum: 49_152 }),
                ));
                target.with_maximum_shared_memory_bytes(65_536).unwrap()
            } else {
                target
            };
            let module = Compiler
                .module(&context, &verified, &target, &Options::default(), &KernelSchedule::default())
                .unwrap();
            assert_eq!(module.verify(), Ok(true));
            let mut operations = Vec::new();
            let mut post_operations = Vec::new();
            module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
                let name = operation.name().as_str().unwrap().to_owned();
                if matches!(name.as_str(), "arith.mulf" | "arith.addf") {
                    post_operations.push(name.clone());
                }
                if matches!(name.as_str(), "nvvm.tcgen05.cp" | "nvvm.tcgen05.mma.block_scale" | "nvvm.tcgen05.commit") {
                    operations.push(name);
                }
                WalkResult::Advance
            });
            assert_eq!(post_operations, if post_scale { vec!["arith.mulf", "arith.addf"] } else { vec![] });
            assert_eq!(
                operations,
                if packed && rows == 128 {
                    vec![
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.commit",
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.commit",
                        "nvvm.tcgen05.mma.block_scale",
                        "nvvm.tcgen05.mma.block_scale",
                        "nvvm.tcgen05.commit",
                    ]
                } else if packed {
                    vec![
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.commit",
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.commit",
                        "nvvm.tcgen05.mma.block_scale",
                        "nvvm.tcgen05.mma.block_scale",
                        "nvvm.tcgen05.commit",
                    ]
                } else if rows == 128 {
                    vec![
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.commit",
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.commit",
                        "nvvm.tcgen05.mma.block_scale",
                        "nvvm.tcgen05.commit",
                    ]
                } else {
                    vec![
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.commit",
                        "nvvm.tcgen05.cp",
                        "nvvm.tcgen05.commit",
                        "nvvm.tcgen05.mma.block_scale",
                        "nvvm.tcgen05.commit",
                    ]
                }
            );
        }
    }
}
