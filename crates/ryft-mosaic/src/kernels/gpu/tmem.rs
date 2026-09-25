//! Explicit tensor-memory lifetimes and fifth-generation tensor-core operations.

use std::borrow::Cow;
use std::fmt::{Display, Formatter};
use std::sync::LazyLock;

use ryft_core::kernels::{KernelExtension, KernelExtensionMemory};
use ryft_core::{
    ArrayIrType, ArrayReferenceTransform, ArrayType, DataType, EffectClasses, Effects, Operation, OperationFormatter,
    ProgramError, ReferenceAccessDescriptor, ReferenceAccessMode, ReferenceAccessOperation, ReferenceEffect,
    ReferenceType, RegionInterface, TypeError,
};

/// A datacenter Blackwell tensor-memory allocation and its ordered asynchronous operations.
///
/// Tensor memory uses canonical reference types for logical contents and lifetime analysis. Its physical placement
/// is established by `Allocate` and retained by the GPU adapter; an ordinary array or scratch reference cannot be
/// substituted for that allocation. An accumulator has 128 rows for one CTA or 256 rows for a two-CTA cooperative
/// program.
/// Native target admission checks that the selected launch geometry implements those logical rows.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TmemOperation {
    /// Allocates uninitialized block scales with `F8E8M0FNU` or signed `F8E4M3FN` elements. Logical rows are multiples
    /// of 32 in [32, 256]; each row contains `blocks` values. Native scale magnitudes and any allocation-owned sign
    /// storage are physical components of the same reference and become initialized together after a copy wait.
    AllocateScales { rows: u16, blocks: u16, data_type: DataType },

    /// Copies an immutable scale array into the second operand's allocation of the same type, returning a completion
    /// token. Signed E4M3 scale signs are preserved separately from native magnitudes. Padding is zero-filled
    /// physically; it is not part of the logical scale array. Neither physical component grants initialization before
    /// the completion token is committed and awaited.
    CopyScales,

    /// Issues MXFP8 matrix multiplication using F8E4M3FN operands, unsigned exponent-only block32 scale references,
    /// and the fifth operand's FP32 destination. Scale references have shapes `[M, K / 32]` and `[N, K / 32]`, with
    /// `M` equal to 128 or 256. Native multiplication and FP32 accumulation define rounding; no BF16 prescaling or
    /// scalar fallback is implied.
    MmaBlockScaled { accumulate: bool },

    /// Issues packed E2M1 matrix multiplication using signed E4M3FN block16 scale references. Inputs are packed U8
    /// arrays `[M, K / 2]` and `[N, K / 2]`, scale references `[M, K / 16]` and `[N, K / 16]`, and an FP32 destination
    /// reference `[M, N]`. Negative scale signs move into the packed multiplicands without changing their magnitude.
    /// Within each byte, the low nibble stores the earlier contraction element. `M` is 128 or 256, `N` is a power of
    /// two in [32, 256], and `K` is positive and divisible by 64. Native multiplication and FP32 accumulation define
    /// rounding. `accumulate` requires an already initialized destination; tensor-wide scales and addends use
    /// ordinary array operations after load.
    MmaNvfp4 { accumulate: bool },

    /// Allocates an uninitialized FP32 tensor-memory matrix with 128 or 256 rows and a power-of-two column count in
    /// 32..=512. Each participating CTA physically owns 128 rows.
    Allocate { rows: u16, columns: u16 },

    /// Issues F16 or BF16 matrix multiplication into the third operand's FP32 tensor-memory reference. The first two
    /// operands are immutable arrays shaped `[M, K]` and `[K, N]`, with `M` equal to 128 or 256. Returns a completion
    /// token reference. With `accumulate`, the destination must already be initialized and its values are added to
    /// the product.
    Mma { accumulate: bool },

    /// Associates the pending multiplication with its completion barrier without publishing destination writes.
    Commit,

    /// Waits for a committed multiplication, publishes its writes, and consumes its completion-token reference.
    Wait,

    /// Loads the initialized tensor-memory reference into an ordinary FP32 array, waiting for native loads to finish.
    Load,

    /// Releases a tensor-memory allocation after all its outstanding operations have completed.
    Release,
}

impl Display for TmemOperation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for TmemOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        match self {
            Self::AllocateScales { .. } => "mosaic_gpu.tmem.allocate_scales",
            Self::CopyScales => "mosaic_gpu.tmem.copy_scales",
            Self::MmaBlockScaled { .. } => "mosaic_gpu.tmem.mma_block_scaled",
            Self::MmaNvfp4 { .. } => "mosaic_gpu.tmem.mma_nvfp4",
            Self::Allocate { .. } => "mosaic_gpu.tmem.allocate",
            Self::Mma { .. } => "mosaic_gpu.tmem.mma",
            Self::Commit => "mosaic_gpu.tmem.commit",
            Self::Wait => "mosaic_gpu.tmem.wait",
            Self::Load => "mosaic_gpu.tmem.load",
            Self::Release => "mosaic_gpu.tmem.release",
        }
    }

    fn infer_output_types(
        &self,
        inputs: &[ArrayIrType],
        regions: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        if !regions.is_empty() {
            return Err(TypeError::invalid(format!("`{}` does not accept regions", self.name())));
        }
        let count = self.base_input_count();
        if inputs.len() != count {
            return Err(TypeError::invalid(format!("`{}` requires {count} inputs", self.name())));
        }
        let token = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Token)));
        match self {
            Self::AllocateScales { rows, blocks, data_type } => {
                if !matches!(data_type, DataType::F8E8M0FNU | DataType::F8E4M3FN) {
                    return Err(TypeError::invalid("tensor-memory scale storage requires F8E8M0FNU or F8E4M3FN"));
                }
                if !(32..=256).contains(rows) || rows % 32 != 0 || *blocks == 0 || *blocks > 128 {
                    return Err(TypeError::invalid(
                        "tensor-memory scales require rows divisible by 32 in [32, 256] and blocks in [1, 128]",
                    ));
                }
                Ok(vec![
                    ReferenceType::new(ArrayType::new_static(*data_type, [usize::from(*rows), usize::from(*blocks)]))
                        .into(),
                ])
            }
            Self::CopyScales => {
                let ArrayIrType::Array(source) = &inputs[0] else {
                    return Err(TypeError::invalid("tensor-memory scale copy requires an immutable array source"));
                };
                let destination = <&ReferenceType<ArrayType>>::try_from(&inputs[1])?.referent();
                if source != destination || !matches!(source.data_type(), DataType::F8E8M0FNU | DataType::F8E4M3FN) {
                    return Err(TypeError::invalid(
                        "tensor-memory scale copy requires identical F8E8M0FNU or F8E4M3FN array and reference types",
                    ));
                }
                Ok(vec![token])
            }
            Self::MmaBlockScaled { .. } | Self::MmaNvfp4 { .. } => {
                let (ArrayIrType::Array(left), ArrayIrType::Array(right)) = (&inputs[0], &inputs[1]) else {
                    return Err(TypeError::invalid(
                        "block-scaled tensor-memory multiplication requires immutable array operands",
                    ));
                };
                let scale_left = <&ReferenceType<ArrayType>>::try_from(&inputs[2])?.referent();
                let scale_right = <&ReferenceType<ArrayType>>::try_from(&inputs[3])?.referent();
                let destination = <&ReferenceType<ArrayType>>::try_from(&inputs[4])?.referent();
                let shapes = [left, right, scale_left, scale_right, destination]
                    .map(|array| array.static_shape().map(|shape| shape.dimensions().to_vec()));
                let packed = matches!(self, Self::MmaNvfp4 { .. });
                let geometry = match &shapes {
                    [Some(left), Some(right), ..] if left.len() == 2 && right.len() == 2 => {
                        left[1].checked_mul(if packed { 2 } else { 1 }).map(|contraction| {
                            (
                                left[0],
                                contraction,
                                right[if packed { 0 } else { 1 }],
                                right[if packed { 1 } else { 0 }] == left[1],
                            )
                        })
                    }
                    _ => None,
                };
                let data_type = if packed { DataType::U8 } else { DataType::F8E4M3FN };
                let scale_type = if packed { DataType::F8E4M3FN } else { DataType::F8E8M0FNU };
                let valid = geometry.is_some_and(|(rows, contraction, columns, compatible)| {
                    compatible
                        && matches!(rows, 128 | 256)
                        && contraction > 0
                        && contraction % if packed { 64 } else { 32 } == 0
                        && (32..=256).contains(&columns)
                        && columns.is_power_of_two()
                        && shapes[2].as_deref() == Some(&[rows, contraction / if packed { 16 } else { 32 }])
                        && shapes[3].as_deref() == Some(&[columns, contraction / if packed { 16 } else { 32 }])
                        && shapes[4].as_deref() == Some(&[rows, columns])
                });
                if left.data_type() != data_type
                    || right.data_type() != data_type
                    || scale_left.data_type() != scale_type
                    || scale_right.data_type() != scale_type
                    || destination.data_type() != DataType::F32
                    || !valid
                {
                    return Err(TypeError::invalid(if packed {
                        concat!(
                            "nvfp4 tensor-memory multiplication requires packed U8 [M, K / 2] and [N, K / 2] arrays, ",
                            "F8E4M3FN block16 scales, and an F32 [M, N] destination, with M in {128, 256}, ",
                            "positive K divisible by 64, and power-of-two N in [32, 256]"
                        )
                    } else {
                        concat!(
                            "block-scaled tensor-memory multiplication requires F8E4M3FN [M, K] and [K, N] arrays, ",
                            "F8E8M0FNU block32 scales, and an F32 [M, N] destination, with M in {128, 256}, ",
                            "positive K divisible by 32, and power-of-two N in [32, 256]"
                        )
                    }));
                }
                Ok(vec![token])
            }
            Self::Allocate { rows, columns } => {
                if !matches!(*rows, 128 | 256) {
                    return Err(TypeError::invalid("tensor-memory accumulator rows must be 128 or 256"));
                }
                if !(32..=512).contains(columns) || !columns.is_power_of_two() {
                    return Err(TypeError::invalid("tensor-memory columns must be a power of two in [32, 512]"));
                }
                Ok(vec![
                    ReferenceType::new(ArrayType::new_static(
                        DataType::F32,
                        [usize::from(*rows), usize::from(*columns)],
                    ))
                    .into(),
                ])
            }
            Self::Commit | Self::Wait => {
                if inputs[0] != token {
                    return Err(TypeError::invalid(format!("`{}` requires a completion-token reference", self.name())));
                }
                Ok(vec![])
            }
            Self::Mma { .. } => {
                let ArrayIrType::Array(left) = &inputs[0] else {
                    return Err(TypeError::invalid("tensor-memory multiplication requires immutable array operands"));
                };
                let ArrayIrType::Array(right) = &inputs[1] else {
                    return Err(TypeError::invalid("tensor-memory multiplication requires immutable array operands"));
                };
                let destination = <&ReferenceType<ArrayType>>::try_from(&inputs[2])?.referent();
                let left_shape = left.static_shape().map(|shape| shape.dimensions().to_vec());
                let right_shape = right.static_shape().map(|shape| shape.dimensions().to_vec());
                let destination_shape = destination.static_shape().map(|shape| shape.dimensions().to_vec());
                if !matches!(left.data_type(), DataType::F16 | DataType::BF16)
                    || left.data_type() != right.data_type()
                    || destination.data_type() != DataType::F32
                    || !matches!((&left_shape, &right_shape, &destination_shape),
                        (Some(left), Some(right), Some(destination))
                        if left.len() == 2 && right.len() == 2 && destination.len() == 2
                            && matches!(left[0], 128 | 256) && left[1] > 0 && left[1] % 16 == 0
                            && right[0] == left[1] && right[1] >= 32 && right[1] <= 256
                            && right[1].is_power_of_two() && destination == &[left[0], right[1]])
                {
                    return Err(TypeError::invalid(concat!(
                        "tensor-memory multiplication requires F16 or BF16 [M, K] and [K, N] arrays ",
                        "and an F32 [M, N] reference, with M in {128, 256}, positive K divisible by 16 ",
                        "and power-of-two N in [32, 256]",
                    )));
                }
                Ok(vec![token])
            }
            Self::Load | Self::Release => {
                let reference = <&ReferenceType<ArrayType>>::try_from(&inputs[0])?;
                let array = reference.referent();
                if matches!(self, Self::Release)
                    && matches!(array.data_type(), DataType::F8E8M0FNU | DataType::F8E4M3FN)
                {
                    return Ok(vec![]);
                }
                if array.data_type() != DataType::F32
                    || !array.static_shape().is_some_and(|shape| {
                        let dimensions = shape.dimensions();
                        dimensions.len() == 2
                            && matches!(dimensions[0], 128 | 256)
                            && (32..=512).contains(&dimensions[1])
                            && dimensions[1].is_power_of_two()
                    })
                {
                    return Err(TypeError::invalid(
                        "tensor-memory access requires an F32 [M, columns] reference with M in {128, 256}",
                    ));
                }
                Ok(if matches!(self, Self::Load) { vec![array.clone().into()] } else { vec![] })
            }
        }
    }

    fn effects(&self) -> Cow<'_, Effects> {
        // Every variant declares one of a few fixed effect lists. Each list is built once, so that the per-input
        // descriptor queries issued by layout validation and lowering do not allocate.
        fn declare(effects: Vec<ReferenceEffect>) -> Effects {
            Effects::new(EffectClasses::NONE, effects).unwrap()
        }
        static ALLOCATE: LazyLock<Effects> =
            LazyLock::new(|| declare(vec![ReferenceEffect::Allocate { output_index: 0 }]));
        static COPY_SCALES: LazyLock<Effects> = LazyLock::new(|| {
            declare(vec![
                ReferenceEffect::Access { input_index: 1, mode: ReferenceAccessMode::Write },
                ReferenceEffect::Allocate { output_index: 0 },
            ])
        });
        // Multiplication lists are indexed by `accumulate`: accumulating multiplications also read their accumulator.
        static SCALED_MMA: LazyLock<[Effects; 2]> = LazyLock::new(|| {
            [ReferenceAccessMode::Write, ReferenceAccessMode::ReadWrite].map(|mode| {
                declare(vec![
                    ReferenceEffect::Access { input_index: 2, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 3, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 4, mode },
                    ReferenceEffect::Allocate { output_index: 0 },
                ])
            })
        });
        static MMA: LazyLock<[Effects; 2]> = LazyLock::new(|| {
            [ReferenceAccessMode::Write, ReferenceAccessMode::ReadWrite].map(|mode| {
                declare(vec![
                    ReferenceEffect::Access { input_index: 2, mode },
                    ReferenceEffect::Allocate { output_index: 0 },
                ])
            })
        });
        static READ: LazyLock<Effects> = LazyLock::new(|| {
            declare(vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }])
        });
        static CONSUME: LazyLock<Effects> = LazyLock::new(|| {
            declare(vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Consume }])
        });
        Cow::Borrowed(match self {
            Self::AllocateScales { .. } | Self::Allocate { .. } => &ALLOCATE,
            Self::CopyScales => &COPY_SCALES,
            Self::MmaBlockScaled { accumulate } | Self::MmaNvfp4 { accumulate } => {
                &SCALED_MMA[usize::from(*accumulate)]
            }
            Self::Mma { accumulate } => &MMA[usize::from(*accumulate)],
            Self::Commit | Self::Load => &READ,
            Self::Wait | Self::Release => &CONSUME,
        })
    }

    fn render(&self, formatter: &mut Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::AllocateScales { rows, blocks, data_type } => {
                OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
                    operation.field("rows", rows)?;
                    operation.field("blocks", blocks)?;
                    operation.field("data_type", data_type)
                })
            }
            Self::Allocate { rows, columns } => OperationFormatter::new(formatter, indentation, self.name())?
                .bracketed(|operation| {
                    operation.field("rows", rows)?;
                    operation.field("columns", columns)
                }),
            Self::MmaBlockScaled { accumulate } | Self::MmaNvfp4 { accumulate } | Self::Mma { accumulate } => {
                OperationFormatter::new(formatter, indentation, self.name())?
                    .bracketed(|operation| operation.field("accumulate", accumulate))
            }
            _ => formatter.write_str(self.name()),
        }
    }
}

impl ReferenceAccessOperation for TmemOperation {
    type Transform = ArrayReferenceTransform;

    fn base_input_count(&self) -> usize {
        match self {
            Self::AllocateScales { .. } | Self::Allocate { .. } => 0,
            Self::CopyScales => 2,
            Self::MmaBlockScaled { .. } | Self::MmaNvfp4 { .. } => 5,
            Self::Mma { .. } => 3,
            _ => 1,
        }
    }

    fn reference_access_descriptor(
        &self,
        input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, Self::Transform>> {
        // Tensor-memory accesses take whole references, so every access has an empty path and no bindings.
        let count = self.base_input_count();
        self.effects()
            .accesses()
            .any(|(index, _)| index == input_index)
            .then(|| ReferenceAccessDescriptor::new(&[], count..count))
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<Self::Transform>,
    ) -> Result<Self, ProgramError> {
        if transforms.is_empty() && self.reference_access_descriptor(input_index).is_some() {
            return Ok(*self);
        }
        Err(ProgramError::UnsupportedOperation {
            message: "tensor-memory operations require whole references".to_owned(),
        })
    }
}

impl KernelExtension for TmemOperation {
    fn memory_semantics(&self) -> Result<KernelExtensionMemory, TypeError> {
        Ok(match self {
            Self::AllocateScales { .. } | Self::Allocate { .. } => {
                KernelExtensionMemory::Allocation { output_index: 0 }
            }
            Self::CopyScales | Self::MmaBlockScaled { .. } | Self::MmaNvfp4 { .. } | Self::Mma { .. } => {
                KernelExtensionMemory::Asynchronous { completion_output_index: 0 }
            }
            Self::Commit => KernelExtensionMemory::Commit { completion_input_index: 0 },
            Self::Wait => KernelExtensionMemory::Wait { completion_input_index: 0 },
            Self::Load => KernelExtensionMemory::Synchronous,
            Self::Release => KernelExtensionMemory::Release { input_index: 0 },
        })
    }

    fn semantic_key(&self) -> Result<Vec<u8>, TypeError> {
        let mut key = b"ryft-mosaic.gpu.tmem.v2\0".to_vec();
        match self {
            Self::AllocateScales { rows, blocks, data_type } => {
                key.push(6);
                key.push(match data_type {
                    DataType::F8E8M0FNU => 0,
                    DataType::F8E4M3FN => 1,
                    _ => return Err(TypeError::invalid("tensor-memory scale storage requires F8E8M0FNU or F8E4M3FN")),
                });
                key.extend_from_slice(&rows.to_le_bytes());
                key.extend_from_slice(&blocks.to_le_bytes());
            }
            Self::CopyScales => key.push(7),
            Self::MmaBlockScaled { accumulate } => key.extend_from_slice(&[8, u8::from(*accumulate)]),
            Self::MmaNvfp4 { accumulate } => key.extend_from_slice(&[9, u8::from(*accumulate)]),
            Self::Allocate { rows, columns } => {
                key.push(0);
                key.extend_from_slice(&rows.to_le_bytes());
                key.extend_from_slice(&columns.to_le_bytes());
            }
            Self::Mma { accumulate } => key.extend_from_slice(&[1, u8::from(*accumulate)]),
            Self::Commit => key.push(2),
            Self::Wait => key.push(3),
            Self::Load => key.push(4),
            Self::Release => key.push(5),
        }
        Ok(key)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_tmem_operation_type_inference() {
        let destination = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [128, 32])));
        let token = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Token)));
        assert_eq!(
            TmemOperation::Allocate { rows: 128, columns: 32 }.infer_output_types(&[], &[]),
            Ok(vec![destination.clone()])
        );
        let inputs = [
            ArrayIrType::Array(ArrayType::new_static(DataType::BF16, [128, 16])),
            ArrayIrType::Array(ArrayType::new_static(DataType::BF16, [16, 32])),
            destination.clone(),
        ];
        assert_eq!(TmemOperation::Mma { accumulate: false }.infer_output_types(&inputs, &[]), Ok(vec![token.clone()]));
        assert_eq!(TmemOperation::Mma { accumulate: true }.infer_output_types(&inputs, &[]), Ok(vec![token.clone()]));
        assert_eq!(TmemOperation::Commit.infer_output_types(&[token.clone()], &[]), Ok(vec![]));
        assert_eq!(TmemOperation::Wait.infer_output_types(&[token], &[]), Ok(vec![]));
        assert_eq!(
            TmemOperation::Load.infer_output_types(&[destination.clone()], &[]),
            Ok(vec![ArrayIrType::Array(ArrayType::new_static(DataType::F32, [128, 32]))])
        );
        assert_eq!(TmemOperation::Release.infer_output_types(&[destination], &[]), Ok(vec![]));
        assert!(matches!(TmemOperation::Allocate { rows: 128, columns: 48 }.infer_output_types(&[], &[]),
            Err(TypeError::Invalid { message }) if message == "tensor-memory columns must be a power of two in [32, 512]"));
        assert!(matches!(TmemOperation::Wait.infer_output_types(&inputs[..1], &[]),
            Err(TypeError::Invalid { message }) if message == "`mosaic_gpu.tmem.wait` requires a completion-token reference"));
    }

    #[test]
    fn test_tmem_operation_type_inference_scales() {
        let scale_left = ArrayType::new_static(DataType::F8E8M0FNU, [128, 1]);
        let scale_right = ArrayType::new_static(DataType::F8E8M0FNU, [32, 1]);
        let allocation = TmemOperation::AllocateScales { rows: 128, blocks: 1, data_type: DataType::F8E8M0FNU };
        assert_eq!(allocation.infer_output_types(&[], &[]), Ok(vec![ReferenceType::new(scale_left.clone()).into()]));
        let token = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Token)));
        assert_eq!(
            TmemOperation::CopyScales
                .infer_output_types(&[scale_left.clone().into(), ReferenceType::new(scale_left.clone()).into(),], &[]),
            Ok(vec![token.clone()])
        );
        let mut inputs = vec![
            ArrayType::new_static(DataType::F8E4M3FN, [128, 32]).into(),
            ArrayType::new_static(DataType::F8E4M3FN, [32, 32]).into(),
            ReferenceType::new(scale_left).into(),
            ReferenceType::new(scale_right.clone()).into(),
            ReferenceType::new(ArrayType::new_static(DataType::F32, [128, 32])).into(),
        ];
        assert_eq!(
            TmemOperation::MmaBlockScaled { accumulate: false }.infer_output_types(&inputs, &[]),
            Ok(vec![token])
        );
        inputs[3] = ReferenceType::new(ArrayType::new_static(DataType::F8E4M3FN, [32, 1])).into();
        assert!(matches!(
            TmemOperation::MmaBlockScaled { accumulate: false }.infer_output_types(&inputs, &[]),
            Err(TypeError::Invalid { message }) if message == concat!(
                "block-scaled tensor-memory multiplication requires F8E4M3FN [M, K] and [K, N] arrays, ",
                "F8E8M0FNU block32 scales, and an F32 [M, N] destination, with M in {128, 256}, ",
                "positive K divisible by 32, and power-of-two N in [32, 256]",
            )
        ));
        assert_eq!(
            TmemOperation::Release.infer_output_types(&[ReferenceType::new(scale_right).into()], &[]),
            Ok(vec![])
        );
        assert_eq!(
            TmemOperation::CopyScales.memory_semantics(),
            Ok(KernelExtensionMemory::Asynchronous { completion_output_index: 0 })
        );
        assert_eq!(
            TmemOperation::MmaBlockScaled { accumulate: false }.effects().reference_effects(),
            &[
                ReferenceEffect::Access { input_index: 2, mode: ReferenceAccessMode::Read },
                ReferenceEffect::Access { input_index: 3, mode: ReferenceAccessMode::Read },
                ReferenceEffect::Access { input_index: 4, mode: ReferenceAccessMode::Write },
                ReferenceEffect::Allocate { output_index: 0 },
            ]
        );
        assert_ne!(
            allocation.semantic_key().unwrap(),
            TmemOperation::AllocateScales { rows: 128, blocks: 2, data_type: DataType::F8E8M0FNU }
                .semantic_key()
                .unwrap()
        );
        assert_ne!(
            TmemOperation::CopyScales.semantic_key().unwrap(),
            TmemOperation::MmaBlockScaled { accumulate: false }.semantic_key().unwrap()
        );
    }

    #[test]
    fn test_tmem_operation_type_inference_nvfp4() {
        let allocation = TmemOperation::AllocateScales { rows: 128, blocks: 8, data_type: DataType::F8E4M3FN };
        assert_eq!(
            allocation.infer_output_types(&[], &[]),
            Ok(vec![ReferenceType::new(ArrayType::new_static(DataType::F8E4M3FN, [128, 8])).into(),])
        );
        let signed = ArrayType::new_static(DataType::F8E4M3FN, [128, 8]);
        assert_eq!(
            TmemOperation::CopyScales
                .infer_output_types(&[signed.clone().into(), ReferenceType::new(signed.clone()).into()], &[]),
            Ok(vec![ReferenceType::new(ArrayType::scalar(DataType::Token)).into()]),
        );
        assert!(matches!(
            TmemOperation::CopyScales.infer_output_types(&[
                signed.into(), ReferenceType::new(ArrayType::new_static(DataType::F8E8M0FNU, [128, 8])).into(),
            ], &[]),
            Err(TypeError::Invalid { message }) if message == "tensor-memory scale copy requires identical F8E8M0FNU or F8E4M3FN array and reference types",
        ));
        let mut inputs = vec![
            ArrayType::new_static(DataType::U8, [128, 64]).into(),
            ArrayType::new_static(DataType::U8, [32, 64]).into(),
            ReferenceType::new(ArrayType::new_static(DataType::F8E4M3FN, [128, 8])).into(),
            ReferenceType::new(ArrayType::new_static(DataType::F8E4M3FN, [32, 8])).into(),
            ReferenceType::new(ArrayType::new_static(DataType::F32, [128, 32])).into(),
        ];
        let operation = TmemOperation::MmaNvfp4 { accumulate: true };
        assert_eq!(
            operation.infer_output_types(&inputs, &[]),
            Ok(vec![ReferenceType::new(ArrayType::scalar(DataType::Token)).into(),])
        );
        let diagnostic = concat!(
            "nvfp4 tensor-memory multiplication requires packed U8 [M, K / 2] and [N, K / 2] arrays, ",
            "F8E4M3FN block16 scales, and an F32 [M, N] destination, with M in {128, 256}, ",
            "positive K divisible by 64, and power-of-two N in [32, 256]",
        );
        inputs[1] = ArrayType::new_static(DataType::U8, [64, 32]).into();
        assert!(
            matches!(operation.infer_output_types(&inputs, &[]), Err(TypeError::Invalid { message }) if message == diagnostic)
        );
        inputs[1] = ArrayType::new_static(DataType::U8, [32, 64]).into();
        inputs[3] = ReferenceType::new(ArrayType::new_static(DataType::F8E8M0FNU, [32, 8])).into();
        assert!(
            matches!(operation.infer_output_types(&inputs, &[]), Err(TypeError::Invalid { message }) if message == diagnostic)
        );
        assert_eq!(
            operation.effects().reference_effects(),
            &[
                ReferenceEffect::Access { input_index: 2, mode: ReferenceAccessMode::Read },
                ReferenceEffect::Access { input_index: 3, mode: ReferenceAccessMode::Read },
                ReferenceEffect::Access { input_index: 4, mode: ReferenceAccessMode::ReadWrite },
                ReferenceEffect::Allocate { output_index: 0 },
            ]
        );
        assert_ne!(
            allocation.semantic_key().unwrap(),
            TmemOperation::AllocateScales { rows: 128, blocks: 8, data_type: DataType::F8E8M0FNU }
                .semantic_key()
                .unwrap()
        );
        assert_ne!(
            operation.semantic_key().unwrap(),
            TmemOperation::MmaBlockScaled { accumulate: true }.semantic_key().unwrap()
        );
    }

    #[test]
    fn test_tmem_operation_memory_semantics() {
        assert_eq!(
            TmemOperation::Allocate { rows: 128, columns: 64 }.memory_semantics(),
            Ok(KernelExtensionMemory::Allocation { output_index: 0 })
        );
        assert_eq!(
            TmemOperation::Mma { accumulate: true }.memory_semantics(),
            Ok(KernelExtensionMemory::Asynchronous { completion_output_index: 0 })
        );
        assert_eq!(
            TmemOperation::Commit.memory_semantics(),
            Ok(KernelExtensionMemory::Commit { completion_input_index: 0 })
        );
        assert_eq!(
            TmemOperation::Wait.memory_semantics(),
            Ok(KernelExtensionMemory::Wait { completion_input_index: 0 })
        );
        assert_eq!(TmemOperation::Load.memory_semantics(), Ok(KernelExtensionMemory::Synchronous));
        assert_eq!(TmemOperation::Release.memory_semantics(), Ok(KernelExtensionMemory::Release { input_index: 0 }));
        assert_eq!(
            TmemOperation::Mma { accumulate: true }.effects().reference_effects(),
            &[
                ReferenceEffect::Access { input_index: 2, mode: ReferenceAccessMode::ReadWrite },
                ReferenceEffect::Allocate { output_index: 0 },
            ]
        );
        assert_eq!(
            TmemOperation::Mma { accumulate: false }.effects().reference_effects(),
            &[
                ReferenceEffect::Access { input_index: 2, mode: ReferenceAccessMode::Write },
                ReferenceEffect::Allocate { output_index: 0 },
            ]
        );
    }

    #[test]
    fn test_tmem_operation_semantic_key() {
        let allocate = TmemOperation::Allocate { rows: 128, columns: 32 };
        let different = TmemOperation::Allocate { rows: 128, columns: 64 };
        assert_ne!(allocate.semantic_key().unwrap(), different.semantic_key().unwrap());
        assert_ne!(
            TmemOperation::Mma { accumulate: false }.semantic_key().unwrap(),
            TmemOperation::Mma { accumulate: true }.semantic_key().unwrap()
        );
        assert_eq!(TmemOperation::Commit.semantic_key(), Ok(b"ryft-mosaic.gpu.tmem.v2\0\x02".to_vec()));
        let values = HashMap::from([(allocate, 1), (different, 2)]);
        assert_eq!(values[&TmemOperation::Allocate { rows: 128, columns: 32 }], 1);
        assert_eq!(allocate.to_string(), "mosaic_gpu.tmem.allocate [rows=128, columns=32]");
        assert_eq!(format!("{allocate:?}"), "Allocate { rows: 128, columns: 32 }");
    }
}
