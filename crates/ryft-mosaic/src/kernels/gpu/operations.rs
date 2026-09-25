//! Exact GPU operations carried by the canonical kernel extension family.

use std::borrow::Cow;
use std::fmt::{Display, Formatter};

use ryft_core::kernels::{KernelExtension, KernelExtensionMemory, NoKernelExtension};
use ryft_core::{
    ArrayIrType, ArrayReferenceView, ArrayType, Context, DataType, DotOperation, EffectClass, EffectClasses, Effects,
    Operation, OperationFormatter, ProgramError, ReferenceViewOperation, ReferenceViewValidationError, RegionInterface,
    TypeError,
};

use crate::kernels::gpu::tmem::TmemOperation;

/// Operations whose instruction selection is part of the kernel contract.
///
/// These operations use canonical array values and Ryft's ordinary staging machinery. Their presence requires an
/// adapter that implements the exact instruction family; it never authorizes a scalar fallback. Target admission
/// checks architecture, participating threads, operand geometry, and storage before constructing native code.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GpuOperation {
    /// Explicit datacenter Blackwell tensor-memory allocation, asynchronous arithmetic, and lifetime operations.
    Tmem(TmemOperation),

    /// Hopper warpgroup matrix multiplication of two rank-two F16 or BF16 arrays, with FP32 accumulation and result.
    /// The left operand has shape `[64, K]`; the right has shape `[K, N]`. Target admission additionally requires
    /// positive `K` divisible by 16, positive `N` divisible by 8 and no larger than 256, and one 128-thread warpgroup.
    /// The accumulator starts at zero. Numerical accumulation follows the native WGMMA instruction contract.
    Wgmma,

    /// Dense Blackwell NVFP4 multiplication with FP32 accumulation, tensor scaling, and an FP32 addend.
    ///
    /// Inputs are packed `U8` arrays `[M, K / 2]` and `[N, K / 2]`, signed `F8E4M3FN` block scales
    /// `[M, K / 16]` and `[N, K / 16]`, and an `F32` addend `[M, N]`. Each byte holds two E2M1 values,
    /// with the earlier contraction element in the low nibble. The right operand and its scales are stored
    /// transposed. Every bit pattern in the packed arrays denotes an E2M1 value, including signed zero.
    ///
    /// A scale applies to 16 consecutive contraction elements. Its sign is folded into those E2M1 values before
    /// native unsigned-scale multiplication; NaN scales remain NaNs. The native product uses FP32 accumulation.
    /// The result is `tensor_scale * product + addend`, with separate FP32 multiplication and addition after the
    /// native product. Omitting the tensor scale means exactly `1.0f32`. Native accumulation order and rounding
    /// apply; no input quantization, saturation, sparsity, or conversion of the result is implicit.
    ///
    /// Target admission requires complete native instruction tiles and the exact block-scaled instruction feature.
    Nvfp4 {
        /// Whether a sixth input supplies a scalar FP32 tensor scale.
        tensor_scale: bool,
    },

    /// Pair-wise sparse Blackwell NVFP4 multiplication with explicit ordered pair metadata.
    ///
    /// Inputs are compressed packed `U8` A `[M, K / 4]`, transposed packed `U8` B `[N, K / 2]`, signed
    /// `F8E4M3FN` scales `[M, K / 32]` and `[N, K / 32]`, an `F32` addend `[M, N]`, and `U8` metadata
    /// `[M, K / 8]`. Each logical eight-element A chunk retains two ordered adjacent pairs (pair-wise 4:8
    /// sparsity). Its metadata byte is exactly one of `0x4`, `0x8`, `0x9`, `0xc`, `0xd`, or `0xe`; the low
    /// two bits select the first pair and the next two bits select the second pair. Compressed values retain
    /// their original order, with earlier elements in the low nibble. Omitted pairs denote zero.
    ///
    /// Each scale spans 32 logical contraction elements, including 16 retained A values. Signed scale handling,
    /// FP32 product accumulation, and final tensor multiplication/addition follow [`Self::Nvfp4`]. This scale
    /// span differs from the dense operation's 16 logical elements. A seventh scalar `F32` input supplies the
    /// optional tensor scale. Native admission requires complete 16-by-8-by-128 tiles on the exact supported
    /// architecture. Invalid runtime metadata raises an ordered assertion before any native sparse instruction.
    Nvfp4Sparse {
        /// Whether a seventh input supplies a scalar FP32 tensor scale.
        tensor_scale: bool,
    },
}

impl From<NoKernelExtension> for GpuOperation {
    fn from(extension: NoKernelExtension) -> Self {
        match extension {}
    }
}

impl Display for GpuOperation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for GpuOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        match self {
            Self::Tmem(operation) => operation.name(),
            Self::Wgmma => "mosaic_gpu.wgmma",
            Self::Nvfp4 { .. } => "mosaic_gpu.nvfp4",
            Self::Nvfp4Sparse { .. } => "mosaic_gpu.nvfp4_sparse",
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        if !region_interfaces.is_empty() {
            return Err(TypeError::invalid(format!("`{}` does not accept regions", self.name())));
        }
        match self {
            Self::Tmem(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Wgmma => {
                let inputs = input_types
                    .iter()
                    .map(|input| match input {
                        ArrayIrType::Array(array) => Ok(array.clone()),
                        _ => Err(TypeError::invalid("`mosaic_gpu.wgmma` requires array operands")),
                    })
                    .collect::<Result<Vec<ArrayType>, TypeError>>()?;
                if inputs.len() != 2
                    || inputs.iter().any(|input| input.rank() != 2)
                    || !matches!(inputs[0].data_type(), DataType::F16 | DataType::BF16)
                    || inputs[0].data_type() != inputs[1].data_type()
                {
                    return Err(TypeError::invalid(
                        "`mosaic_gpu.wgmma` requires two rank-two operands with the same F16 or BF16 element type",
                    ));
                }
                DotOperation::matmul()
                    .with_accumulation_type(DataType::F32)
                    .infer_output_types(&inputs, &[])
                    .map(|outputs| outputs.into_iter().map(ArrayIrType::Array).collect())
            }
            Self::Nvfp4 { tensor_scale } | Self::Nvfp4Sparse { tensor_scale } => {
                let sparse = matches!(self, Self::Nvfp4Sparse { .. });
                let count = if sparse { 6 } else { 5 };
                if input_types.len() != count + usize::from(*tensor_scale) {
                    return Err(TypeError::invalid(format!(
                        "`{}` requires {} operands and its declared tensor scale",
                        self.name(),
                        if sparse { "six" } else { "five" },
                    )));
                }
                let inputs = input_types
                    .iter()
                    .map(|input| match input {
                        ArrayIrType::Array(array) => Ok(array),
                        _ => Err(TypeError::invalid(format!("`{}` requires array operands", self.name()))),
                    })
                    .collect::<Result<Vec<_>, TypeError>>()?;
                let shapes = inputs[..count]
                    .iter()
                    .map(|input| {
                        input.static_shape().filter(|shape| shape.dimensions().len() == 2).ok_or_else(|| {
                            TypeError::invalid(format!("`{}` requires static rank-two matrix operands", self.name()))
                        })
                    })
                    .collect::<Result<Vec<_>, TypeError>>()?;
                let left = shapes[0].dimensions();
                let right = shapes[1].dimensions();
                if inputs[0].data_type() != DataType::U8
                    || inputs[1].data_type() != DataType::U8
                    || inputs[2].data_type() != DataType::F8E4M3FN
                    || inputs[3].data_type() != DataType::F8E4M3FN
                    || inputs[4].data_type() != DataType::F32
                {
                    return Err(TypeError::invalid(format!(
                        "`{}` requires packed U8 operands, F8E4M3FN scales, and an F32 addend",
                        self.name(),
                    )));
                }
                if sparse && inputs[5].data_type() != DataType::U8 {
                    return Err(TypeError::invalid("`mosaic_gpu.nvfp4_sparse` metadata must be a U8 array"));
                }
                if left[1] == 0
                    || left[1] % 8 != 0
                    || left[1].checked_mul(if sparse { 2 } else { 1 }) != Some(right[1])
                    || shapes[2].dimensions() != [left[0], left[1] / 8]
                    || shapes[3].dimensions() != [right[0], right[1] / if sparse { 16 } else { 8 }]
                    || (sparse && shapes[5].dimensions() != [left[0], left[1] / 2])
                    || shapes[4].dimensions() != [left[0], right[0]]
                {
                    return Err(TypeError::invalid(format!(
                        "`{}` operand, block-scale, and addend shapes are incompatible",
                        self.name(),
                    )));
                }
                if *tensor_scale && (inputs[count].data_type() != DataType::F32 || inputs[count].rank() != 0) {
                    return Err(TypeError::invalid(format!(
                        "`{}` tensor scale must be a scalar F32 array",
                        self.name(),
                    )));
                }
                Ok(vec![ArrayIrType::Array(inputs[4].clone())])
            }
        }
    }

    fn effects(&self) -> Cow<'_, Effects> {
        match self {
            Self::Tmem(operation) => operation.effects(),
            Self::Nvfp4Sparse { .. } => {
                Cow::Owned(Effects::explicit(EffectClasses::single(EffectClass::OrderedAssertion)))
            }
            _ => Cow::Borrowed(Effects::empty()),
        }
    }

    fn render(&self, formatter: &mut Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Tmem(operation) => operation.render(formatter, indentation),
            Self::Wgmma => formatter.write_str(self.name()),
            Self::Nvfp4 { tensor_scale } | Self::Nvfp4Sparse { tensor_scale } => {
                OperationFormatter::new(formatter, indentation, self.name())?
                    .bracketed(|operation| operation.field("tensor_scale", tensor_scale))
            }
        }
    }
}

impl ReferenceViewOperation for GpuOperation {
    type View = ArrayReferenceView;

    fn reference_view(&self, _output_index: usize) -> Option<ArrayReferenceView> {
        None
    }

    fn validate_reference_view(
        view: &ArrayReferenceView,
        source: &ArrayIrType,
        target: &ArrayIrType,
    ) -> Result<(), ReferenceViewValidationError> {
        view.validate(source, target)
    }

    fn reapply_reference_view<C: Context<Type = ArrayIrType, Operation = Self>>(
        _context: &C,
        _view: &ArrayReferenceView,
        _source: C::Value,
        _symbols: &[C::Value],
    ) -> Result<C::Value, ProgramError> {
        Err(ProgramError::MalformedProgram("gpu instruction operations do not construct reference views".to_owned()))
    }
}

impl KernelExtension for GpuOperation {
    fn memory_semantics(&self) -> Result<KernelExtensionMemory, TypeError> {
        match self {
            Self::Tmem(operation) => operation.memory_semantics(),
            _ => Ok(KernelExtensionMemory::Synchronous),
        }
    }

    fn semantic_key(&self) -> Result<Vec<u8>, TypeError> {
        match self {
            Self::Tmem(operation) => operation.semantic_key(),
            Self::Wgmma => Ok(b"ryft-mosaic.gpu.operations.v1\0wgmma\0".to_vec()),
            Self::Nvfp4Sparse { tensor_scale } => {
                let mut key = b"ryft-mosaic.gpu.operations.v1\0nvfp4_sparse\0".to_vec();
                key.push(u8::from(*tensor_scale));
                Ok(key)
            }
            Self::Nvfp4 { tensor_scale } => {
                let mut key = b"ryft-mosaic.gpu.operations.v1\0nvfp4\0".to_vec();
                key.push(u8::from(*tensor_scale));
                Ok(key)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_gpu_operation_wgmma() {
        assert_eq!(GpuOperation::Wgmma.name(), "mosaic_gpu.wgmma");
        assert_eq!(GpuOperation::Wgmma.to_string(), "mosaic_gpu.wgmma");
        assert_eq!(format!("{:?}", GpuOperation::Wgmma), "Wgmma");
        assert_eq!(GpuOperation::Wgmma.memory_semantics(), Ok(KernelExtensionMemory::Synchronous));
        assert_eq!(GpuOperation::Wgmma.semantic_key(), Ok(b"ryft-mosaic.gpu.operations.v1\0wgmma\0".to_vec()));
    }

    #[test]
    fn test_gpu_operation_wgmma_type_inference() {
        let inputs = [
            ArrayIrType::Array(ArrayType::new_static(DataType::F16, vec![64, 16])),
            ArrayIrType::Array(ArrayType::new_static(DataType::F16, vec![16, 8])),
        ];
        assert_eq!(
            GpuOperation::Wgmma.infer_output_types(&inputs, &[]),
            Ok(vec![ArrayIrType::Array(ArrayType::new_static(DataType::F32, vec![64, 8]))])
        );
        assert!(matches!(GpuOperation::Wgmma.infer_output_types(&inputs[..1], &[]),
            Err(TypeError::Invalid { message, .. })
                if message == "`mosaic_gpu.wgmma` requires two rank-two operands with the same F16 or BF16 element type"));
    }

    #[test]
    fn test_gpu_operation_nvfp4_sparse_type_inference() {
        let mut inputs = vec![
            ArrayIrType::Array(ArrayType::new_static(DataType::U8, [16, 32])),
            ArrayIrType::Array(ArrayType::new_static(DataType::U8, [8, 64])),
            ArrayIrType::Array(ArrayType::new_static(DataType::F8E4M3FN, [16, 4])),
            ArrayIrType::Array(ArrayType::new_static(DataType::F8E4M3FN, [8, 4])),
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [16, 8])),
            ArrayIrType::Array(ArrayType::new_static(DataType::U8, [16, 16])),
        ];
        let operation = GpuOperation::Nvfp4Sparse { tensor_scale: false };
        assert_eq!(operation.infer_output_types(&inputs, &[]), Ok(vec![inputs[4].clone()]));
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(operation.to_string(), "mosaic_gpu.nvfp4_sparse [tensor_scale=false]");
        assert_ne!(operation.semantic_key(), GpuOperation::Nvfp4 { tensor_scale: false }.semantic_key());
        inputs[5] = ArrayIrType::Array(ArrayType::new_static(DataType::U8, [16, 8]));
        assert!(matches!(operation.infer_output_types(&inputs, &[]), Err(TypeError::Invalid { message })
            if message == "`mosaic_gpu.nvfp4_sparse` operand, block-scale, and addend shapes are incompatible"));
    }

    #[test]
    fn test_gpu_operation_nvfp4_type_inference() {
        let mut inputs = vec![
            ArrayIrType::Array(ArrayType::new_static(DataType::U8, vec![16, 32])),
            ArrayIrType::Array(ArrayType::new_static(DataType::U8, vec![8, 32])),
            ArrayIrType::Array(ArrayType::new_static(DataType::F8E4M3FN, vec![16, 4])),
            ArrayIrType::Array(ArrayType::new_static(DataType::F8E4M3FN, vec![8, 4])),
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, vec![16, 8])),
        ];
        let operation = GpuOperation::Nvfp4 { tensor_scale: false };
        assert_eq!(operation.infer_output_types(&inputs, &[]), Ok(vec![inputs[4].clone()]));
        let scaled = GpuOperation::Nvfp4 { tensor_scale: true };
        assert!(matches!(scaled.infer_output_types(&inputs, &[]), Err(TypeError::Invalid { message })
            if message == "`mosaic_gpu.nvfp4` requires five operands and its declared tensor scale"));
        inputs.push(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(scaled.infer_output_types(&inputs, &[]), Ok(vec![inputs[4].clone()]));
        inputs[5] = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        assert!(matches!(scaled.infer_output_types(&inputs, &[]), Err(TypeError::Invalid { message })
            if message == "`mosaic_gpu.nvfp4` tensor scale must be a scalar F32 array"));
        inputs.pop();
        inputs[3] = ArrayIrType::Array(ArrayType::new_static(DataType::F8E4M3FN, vec![8, 2]));
        assert!(matches!(operation.infer_output_types(&inputs, &[]), Err(TypeError::Invalid { message })
            if message == "`mosaic_gpu.nvfp4` operand, block-scale, and addend shapes are incompatible"));
    }

    #[test]
    fn test_gpu_operation_nvfp4_semantic_key() {
        let operation = GpuOperation::Nvfp4 { tensor_scale: false };
        let scaled = GpuOperation::Nvfp4 { tensor_scale: true };
        assert_eq!(operation.semantic_key(), Ok(b"ryft-mosaic.gpu.operations.v1\0nvfp4\0\0".to_vec()));
        assert_eq!(scaled.semantic_key(), Ok(b"ryft-mosaic.gpu.operations.v1\0nvfp4\0\x01".to_vec()));
        assert_ne!(operation.to_string(), scaled.to_string());
        assert_eq!(operation.memory_semantics(), Ok(KernelExtensionMemory::Synchronous));
    }
}
