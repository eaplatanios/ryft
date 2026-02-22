use ryft_xla_sys::bindings::{
    MlirAttribute, stablehloConvDimensionNumbersGet, stablehloDotAlgorithmGet, stablehloDotDimensionNumbersGet,
};

use crate::{
    ArrayAttributeRef, Attribute, BooleanAttributeRef, Context, DenseBooleanArrayAttributeRef,
    DenseInteger64ArrayAttributeRef, DetachedOp, DialectHandle, Float8Type, Float8TypeRef, FloatType, FloatTypeRef,
    IntegerAttributeRef, IntoWithContext, Location, Operation, OperationBuilder, Type, Value, ValueRef,
    mlir_attribute_field, mlir_enum_attribute, mlir_op, mlir_op_trait, mlir_subtype_trait_impls,
};

use super::{HasPadding, PADDING_ATTRIBUTE};

mlir_enum_attribute!(
    rust_name = Precision,
    mlir_name = Precision,
    description = "StableHLO precision for operations like [`DotGeneralOperation`] and [`ConvolutionOperation`]",
    variants = {
        Default => "DEFAULT",
        High => "HIGH",
        Highest => "HIGHEST",
    },
    rust_prefix = stable_hlo,
    mlir_prefix = stablehlo,
    mlir_dialect_handle_constructor = stable_hlo,
);

/// StableHLO [`Attribute`] that models the dimension information in [`DotGeneralOperation`]s.
#[derive(Copy, Clone)]
pub struct DotDimensionsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> DotDimensionsAttributeRef<'c, 't> {
    mlir_attribute_field!(
        lhs_batching_dimensions,
        DotDimensionNumbersGetLhsBatchingDimensions,
        [usize],
        mlir_prefix = stablehlo
    );

    mlir_attribute_field!(
        rhs_batching_dimensions,
        DotDimensionNumbersGetRhsBatchingDimensions,
        [usize],
        mlir_prefix = stablehlo
    );

    mlir_attribute_field!(
        lhs_contracting_dimensions,
        DotDimensionNumbersGetLhsContractingDimensions,
        [usize],
        mlir_prefix = stablehlo
    );

    mlir_attribute_field!(
        rhs_contracting_dimensions,
        DotDimensionNumbersGetRhsContractingDimensions,
        [usize],
        mlir_prefix = stablehlo
    );
}

mlir_subtype_trait_impls!(
    DotDimensionsAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = DotDimensionNumbers,
    mlir_prefix = stablehlo,
);

impl<'t> Context<'t> {
    /// Creates a new StableHLO [`DotDimensionsAttributeRef`] owned by this [`Context`]. Refer to the documentation of
    /// [`DotGeneralOperation`] for information on the arguments of this function.
    pub fn stable_hlo_dot_dimensions<'c>(
        &'c self,
        lhs_batching_dimensions: &[usize],
        rhs_batching_dimensions: &[usize],
        lhs_contracting_dimensions: &[usize],
        rhs_contracting_dimensions: &[usize],
    ) -> DotDimensionsAttributeRef<'c, 't> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo());
        let lhs_batching_dimensions = lhs_batching_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let rhs_batching_dimensions = rhs_batching_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let lhs_contracting_dimensions = lhs_contracting_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let rhs_contracting_dimensions = rhs_contracting_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            DotDimensionsAttributeRef::from_c_api(
                stablehloDotDimensionNumbersGet(
                    *self.handle.borrow(),
                    lhs_batching_dimensions.len().cast_signed(),
                    lhs_batching_dimensions.as_ptr(),
                    rhs_batching_dimensions.len().cast_signed(),
                    rhs_batching_dimensions.as_ptr(),
                    lhs_contracting_dimensions.len().cast_signed(),
                    lhs_contracting_dimensions.as_ptr(),
                    rhs_contracting_dimensions.len().cast_signed(),
                    rhs_contracting_dimensions.as_ptr(),
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// StableHLO [`Attribute`] that models the algorithm constraints in [`DotGeneralOperation`]s.
#[derive(Copy, Clone)]
pub struct DotAlgorithmAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> DotAlgorithmAttributeRef<'c, 't> {
    mlir_attribute_field!(lhs_precision_type, DotAlgorithmGetLhsPrecisionType, FloatTypeRef, mlir_prefix = stablehlo);

    mlir_attribute_field!(rhs_precision_type, DotAlgorithmGetRhsPrecisionType, FloatTypeRef, mlir_prefix = stablehlo);

    mlir_attribute_field!(accumulation_type, DotAlgorithmGetAccumulationType, FloatTypeRef, mlir_prefix = stablehlo);

    mlir_attribute_field!(lhs_component_count, DotAlgorithmGetLhsComponentCount, usize, mlir_prefix = stablehlo);

    mlir_attribute_field!(rhs_component_count, DotAlgorithmGetRhsComponentCount, usize, mlir_prefix = stablehlo);

    mlir_attribute_field!(
        primitive_operation_count,
        DotAlgorithmGetNumPrimitiveOperations,
        usize,
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        allow_imprecise_accumulation,
        DotAlgorithmGetAllowImpreciseAccumulation,
        bool,
        mlir_prefix = stablehlo,
    );
}

mlir_subtype_trait_impls!(
    DotAlgorithmAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = DotAlgorithm,
    mlir_prefix = stablehlo,
);

/// Preset configurations for StableHLO [`DotAlgorithmAttributeRef`], based on [`jax.lax.DotAlgorithmPreset`](
/// https://github.com/jax-ml/jax/blob/d76e24313e0db4e060d199e940d85a178ae6aebb/jax/_src/lax/lax.py#L2143).
///
/// These presets provide a named collection of commonly supported `dot` algorithms. Support is platform-dependent,
/// and unsupported algorithms should be treated as compilation-time errors rather than as requests that silently fall
/// back to another algorithm.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[derive(Default)]
pub enum DotAlgorithmPreset<'c, 't> {
    /// Default (platform-specific) algorithm based on the input/output types. This preset does not prescribe a concrete
    /// input type, output type, or accumulation type.
    #[default]
    Default,

    /// Accepts any [`Float8Type`] input type and accumulates into [`Float32TypeRef`](crate::Float32TypeRef).
    /// It supports outputs in [`Float16TypeRef`](crate::Float16TypeRef), [`BFloat16TypeRef`](crate::BFloat16TypeRef),
    /// [`Float32TypeRef`](crate::Float32TypeRef), and a subset of [`Float8Type`] types:
    /// [`Float8E4M3FNTypeRef`](crate::Float8E4M3FNTypeRef), [`Float8E4M3FNUZTypeRef`](crate::Float8E4M3FNUZTypeRef),
    /// [`Float8E4M3B11FNUZTypeRef`](crate::Float8E4M3B11FNUZTypeRef), [`Float8E5M2TypeRef`](crate::Float8E5M2TypeRef),
    /// and [`Float8E5M2FNUZTypeRef`](crate::Float8E5M2FNUZTypeRef).
    Any_F8_Any_F8_F32 { lhs_precision_type: Float8TypeRef<'c, 't>, rhs_precision_type: Float8TypeRef<'c, 't> },

    /// Like [`DotAlgorithmPreset::Any_F8_Any_F8_F32`], but with faster/less precise accumulation.
    Any_F8_Any_F8_F32_Fast_Accumulation {
        lhs_precision_type: Float8TypeRef<'c, 't>,
        rhs_precision_type: Float8TypeRef<'c, 't>,
    },

    /// Like [`DotAlgorithmPreset::Any_F8_Any_F8_F32`], but with its accumulation type depending
    /// on higher-level context.
    Any_F8_Any_F8_Any {
        lhs_precision_type: Float8TypeRef<'c, 't>,
        rhs_precision_type: Float8TypeRef<'c, 't>,
        accumulation_type: FloatTypeRef<'c, 't>,
    },

    /// Like [`DotAlgorithmPreset::Any_F8_Any_F8_F32_Fast_Accumulation`], but with its accumulation type depending
    /// on higher-level context.
    Any_F8_Any_F8_Any_Fast_Accumulation {
        lhs_precision_type: Float8TypeRef<'c, 't>,
        rhs_precision_type: Float8TypeRef<'c, 't>,
        accumulation_type: FloatTypeRef<'c, 't>,
    },

    /// Uses [`Float16TypeRef`](crate::Float16TypeRef) input precision and [`Float16TypeRef`](crate::Float16TypeRef)
    /// accumulation and output precision.
    F16_F16_F16,

    /// Uses [`Float16TypeRef`](crate::Float16TypeRef) input precision and [`Float32TypeRef`](crate::Float32TypeRef)
    /// accumulation and output precision, though it also allows for [`Float16TypeRef`](crate::Float16TypeRef) output
    /// when both inputs are effectively [`Float16TypeRef`](crate::Float16TypeRef).
    F16_F16_F32,

    /// Uses [`BFloat16TypeRef`](crate::BFloat16TypeRef) input precision and [`BFloat16TypeRef`](crate::BFloat16TypeRef)
    /// accumulation and output precision.
    BF16_BF16_BF16,

    /// Uses [`BFloat16TypeRef`](crate::BFloat16TypeRef) input precision and [`Float32TypeRef`](crate::Float32TypeRef)
    /// accumulation and output precision, though it also allows for [`BFloat16TypeRef`](crate::BFloat16TypeRef) output
    /// when both inputs are effectively [`BFloat16TypeRef`](crate::BFloat16TypeRef).
    BF16_BF16_F32,

    /// Like [`DotAlgorithmPreset::BF16_BF16_F32`], but decomposing into 3 primitive operations.
    BF16_BF16_F32_X3,

    /// Like [`DotAlgorithmPreset::BF16_BF16_F32_X3`], but decomposing into 6 primitive operations.
    BF16_BF16_F32_X6,

    /// Like [`DotAlgorithmPreset::BF16_BF16_F32_X3`], but decomposing into 9 primitive operations.
    BF16_BF16_F32_X9,

    /// Uses [`FloatTF32TypeRef`](crate::FloatTF32TypeRef) input precision and
    /// [`FloatTF32TypeRef`](crate::FloatTF32TypeRef) accumulation and output precision.
    TF32_TF32_F32,

    /// Like [`DotAlgorithmPreset::TF32_TF32_F32`], but decomposing into 3 primitive operations.
    TF32_TF32_F32_X3,

    /// Uses [`Float32TypeRef`](crate::Float32TypeRef) input precision and [`Float32TypeRef`](crate::Float32TypeRef)
    /// accumulation and output precision.
    F32_F32_F32,

    /// Uses [`Float64TypeRef`](crate::Float64TypeRef) input precision and [`Float64TypeRef`](crate::Float64TypeRef)
    /// accumulation and output precision.
    F64_F64_F64,
}

impl<'c, 't> DotAlgorithmPreset<'c, 't> {
    /// Creates a new [`DotAlgorithmPreset::Any_F8_Any_F8_F32`].
    pub fn any_f8_any_f8_f32<L: Float8Type<'c, 't>, R: Float8Type<'c, 't>>(
        lhs_precision_type: L,
        rhs_precision_type: R,
    ) -> Self {
        Self::Any_F8_Any_F8_F32 {
            lhs_precision_type: lhs_precision_type.as_ref().cast::<Float8TypeRef>().unwrap(),
            rhs_precision_type: rhs_precision_type.as_ref().cast::<Float8TypeRef>().unwrap(),
        }
    }

    /// Creates a new [`DotAlgorithmPreset::Any_F8_Any_F8_F32_Fast_Accumulation`].
    pub fn any_f8_any_f8_f32_fast_accumulation<L: Float8Type<'c, 't>, R: Float8Type<'c, 't>>(
        lhs_precision_type: L,
        rhs_precision_type: R,
    ) -> Self {
        Self::Any_F8_Any_F8_F32_Fast_Accumulation {
            lhs_precision_type: lhs_precision_type.as_ref().cast::<Float8TypeRef>().unwrap(),
            rhs_precision_type: rhs_precision_type.as_ref().cast::<Float8TypeRef>().unwrap(),
        }
    }

    /// Creates a new [`DotAlgorithmPreset::Any_F8_Any_F8_Any`].
    pub fn any_f8_any_f8_any<L: Float8Type<'c, 't>, R: Float8Type<'c, 't>, T: FloatType<'c, 't>>(
        lhs_precision_type: L,
        rhs_precision_type: R,
        accumulation_type: T,
    ) -> Self {
        Self::Any_F8_Any_F8_Any {
            lhs_precision_type: lhs_precision_type.as_ref().cast::<Float8TypeRef>().unwrap(),
            rhs_precision_type: rhs_precision_type.as_ref().cast::<Float8TypeRef>().unwrap(),
            accumulation_type: accumulation_type.as_ref().cast::<FloatTypeRef>().unwrap(),
        }
    }

    /// Creates a new [`DotAlgorithmPreset::Any_F8_Any_F8_Any_Fast_Accumulation`].
    pub fn any_f8_any_f8_any_fast_accumulation<L: Float8Type<'c, 't>, R: Float8Type<'c, 't>, T: FloatType<'c, 't>>(
        lhs_precision_type: L,
        rhs_precision_type: R,
        accumulation_type: T,
    ) -> Self {
        Self::Any_F8_Any_F8_Any_Fast_Accumulation {
            lhs_precision_type: lhs_precision_type.as_ref().cast::<Float8TypeRef>().unwrap(),
            rhs_precision_type: rhs_precision_type.as_ref().cast::<Float8TypeRef>().unwrap(),
            accumulation_type: accumulation_type.as_ref().cast::<FloatTypeRef>().unwrap(),
        }
    }

    /// Creates a new [`DotAlgorithmPreset::F16_F16_F16`].
    pub fn f16_f16_f16() -> Self {
        Self::F16_F16_F16
    }

    /// Creates a new [`DotAlgorithmPreset::F16_F16_F32`].
    pub fn f16_f16_f32() -> Self {
        Self::F16_F16_F32
    }

    /// Creates a new [`DotAlgorithmPreset::BF16_BF16_BF16`].
    pub fn bf16_bf16_bf16() -> Self {
        Self::BF16_BF16_BF16
    }

    /// Creates a new [`DotAlgorithmPreset::BF16_BF16_F32`].
    pub fn bf16_bf16_f32() -> Self {
        Self::BF16_BF16_F32
    }

    /// Creates a new [`DotAlgorithmPreset::BF16_BF16_F32_X3`].
    pub fn bf16_bf16_f32_x3() -> Self {
        Self::BF16_BF16_F32_X3
    }

    /// Creates a new [`DotAlgorithmPreset::BF16_BF16_F32_X6`].
    pub fn bf16_bf16_f32_x6() -> Self {
        Self::BF16_BF16_F32_X6
    }

    /// Creates a new [`DotAlgorithmPreset::BF16_BF16_F32_X9`].
    pub fn bf16_bf16_f32_x9() -> Self {
        Self::BF16_BF16_F32_X9
    }

    /// Creates a new [`DotAlgorithmPreset::TF32_TF32_F32`].
    pub fn tf32_tf32_f32() -> Self {
        Self::TF32_TF32_F32
    }

    /// Creates a new [`DotAlgorithmPreset::TF32_TF32_F32_X3`].
    pub fn tf32_tf32_f32_x3() -> Self {
        Self::TF32_TF32_F32_X3
    }

    /// Creates a new [`DotAlgorithmPreset::F32_F32_F32`].
    pub fn f32_f32_f32() -> Self {
        Self::F32_F32_F32
    }

    /// Creates a new [`DotAlgorithmPreset::F64_F64_F64`].
    pub fn f64_f64_f64() -> Self {
        Self::F64_F64_F64
    }
}

impl<'c, 't> IntoWithContext<'c, 't, Option<DotAlgorithmAttributeRef<'c, 't>>> for DotAlgorithmPreset<'c, 't> {
    fn into_with_context(self, context: &'c Context<'t>) -> Option<DotAlgorithmAttributeRef<'c, 't>> {
        match self {
            DotAlgorithmPreset::Default => None,
            DotAlgorithmPreset::Any_F8_Any_F8_F32 { lhs_precision_type, rhs_precision_type } => {
                Some(context.stable_hlo_dot_algorithm(
                    lhs_precision_type,
                    rhs_precision_type,
                    context.float32_type(),
                    1,
                    1,
                    1,
                    false,
                ))
            }
            DotAlgorithmPreset::Any_F8_Any_F8_F32_Fast_Accumulation { lhs_precision_type, rhs_precision_type } => {
                Some(context.stable_hlo_dot_algorithm(
                    lhs_precision_type,
                    rhs_precision_type,
                    context.float32_type(),
                    1,
                    1,
                    1,
                    true,
                ))
            }
            DotAlgorithmPreset::Any_F8_Any_F8_Any { lhs_precision_type, rhs_precision_type, accumulation_type } => {
                Some(context.stable_hlo_dot_algorithm(
                    lhs_precision_type,
                    rhs_precision_type,
                    accumulation_type,
                    1,
                    1,
                    1,
                    false,
                ))
            }
            DotAlgorithmPreset::Any_F8_Any_F8_Any_Fast_Accumulation {
                lhs_precision_type,
                rhs_precision_type,
                accumulation_type,
            } => Some(context.stable_hlo_dot_algorithm(
                lhs_precision_type,
                rhs_precision_type,
                accumulation_type,
                1,
                1,
                1,
                true,
            )),
            DotAlgorithmPreset::F16_F16_F16 => Some(context.stable_hlo_dot_algorithm(
                context.float16_type(),
                context.float16_type(),
                context.float16_type(),
                1,
                1,
                1,
                false,
            )),
            DotAlgorithmPreset::F16_F16_F32 => Some(context.stable_hlo_dot_algorithm(
                context.float16_type(),
                context.float16_type(),
                context.float32_type(),
                1,
                1,
                1,
                false,
            )),
            DotAlgorithmPreset::BF16_BF16_BF16 => Some(context.stable_hlo_dot_algorithm(
                context.bfloat16_type(),
                context.bfloat16_type(),
                context.bfloat16_type(),
                1,
                1,
                1,
                false,
            )),
            DotAlgorithmPreset::BF16_BF16_F32 => Some(context.stable_hlo_dot_algorithm(
                context.bfloat16_type(),
                context.bfloat16_type(),
                context.float32_type(),
                1,
                1,
                1,
                false,
            )),
            DotAlgorithmPreset::BF16_BF16_F32_X3 => Some(context.stable_hlo_dot_algorithm(
                context.bfloat16_type(),
                context.bfloat16_type(),
                context.float32_type(),
                1,
                1,
                3,
                false,
            )),
            DotAlgorithmPreset::BF16_BF16_F32_X6 => Some(context.stable_hlo_dot_algorithm(
                context.bfloat16_type(),
                context.bfloat16_type(),
                context.float32_type(),
                1,
                1,
                6,
                false,
            )),
            DotAlgorithmPreset::BF16_BF16_F32_X9 => Some(context.stable_hlo_dot_algorithm(
                context.bfloat16_type(),
                context.bfloat16_type(),
                context.float32_type(),
                1,
                1,
                9,
                false,
            )),
            DotAlgorithmPreset::TF32_TF32_F32 => Some(context.stable_hlo_dot_algorithm(
                context.floattf32_type(),
                context.floattf32_type(),
                context.float32_type(),
                1,
                1,
                1,
                false,
            )),
            DotAlgorithmPreset::TF32_TF32_F32_X3 => Some(context.stable_hlo_dot_algorithm(
                context.floattf32_type(),
                context.floattf32_type(),
                context.float32_type(),
                1,
                1,
                3,
                false,
            )),
            DotAlgorithmPreset::F32_F32_F32 => Some(context.stable_hlo_dot_algorithm(
                context.float32_type(),
                context.float32_type(),
                context.float32_type(),
                1,
                1,
                1,
                false,
            )),
            DotAlgorithmPreset::F64_F64_F64 => Some(context.stable_hlo_dot_algorithm(
                context.float64_type(),
                context.float64_type(),
                context.float64_type(),
                1,
                1,
                1,
                false,
            )),
        }
    }
}

impl<'t> Context<'t> {
    /// Creates a new StableHLO [`DotAlgorithmAttributeRef`] owned by this [`Context`]. Refer to the documentation of
    /// [`DotGeneralOperation`] for information on the arguments of this function.
    #[allow(clippy::too_many_arguments)]
    pub fn stable_hlo_dot_algorithm<'c, L: FloatType<'c, 't>, R: FloatType<'c, 't>, T: FloatType<'c, 't>>(
        &'c self,
        lhs_precision_type: L,
        rhs_precision_type: R,
        accumulation_type: T,
        lhs_component_count: usize,
        rhs_component_count: usize,
        primitive_operation_count: usize,
        allow_imprecise_accumulation: bool,
    ) -> DotAlgorithmAttributeRef<'c, 't> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo());
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            DotAlgorithmAttributeRef::from_c_api(
                stablehloDotAlgorithmGet(
                    *self.handle.borrow(),
                    lhs_precision_type.to_c_api(),
                    rhs_precision_type.to_c_api(),
                    accumulation_type.to_c_api(),
                    lhs_component_count as i64,
                    rhs_component_count as i64,
                    primitive_operation_count as i64,
                    allow_imprecise_accumulation,
                ),
                self,
            )
            .unwrap()
        }
    }

    /// Creates an optional StableHLO [`DotAlgorithmAttributeRef`] from the provided [`DotAlgorithmPreset`].
    pub fn stable_hlo_dot_algorithm_from_preset<'c>(
        &'c self,
        preset: DotAlgorithmPreset<'c, 't>,
    ) -> Option<DotAlgorithmAttributeRef<'c, 't>> {
        preset.into_with_context(self)
    }
}

/// Name of the [`Attribute`] that is used to store [`DotGeneralOperation::dimensions`].
pub const DOT_DIMENSIONS_ATTRIBUTE: &str = "dot_dimension_numbers";

/// Name of the [`Attribute`] that is used to store [`DotGeneralOperation::precision`].
pub const DOT_PRECISION_ATTRIBUTE: &str = "precision_config";

/// Name of the [`Attribute`] that is used to store [`DotGeneralOperation::algorithm`].
pub const DOT_ALGORITHM_ATTRIBUTE: &str = "algorithm";

/// StableHLO [`Operation`] that computes the dot product between slices of two tensors and has configurable
/// contracting and batch dimensions. It generalizes matrix multiplication to support batching as well as an arbitrary
/// number of dimension contractions. Specifically, this operation has two input/operand tensors, `lhs` and `rhs`, and
/// one output/result tensor, `result`, where `result[result_index] = dot_product` and:
///
///   - `lhs_result_dims = [d for d in axes(lhs) and d not in lhs_batching_dims and d not in lhs_contracting_dims]`,
///     where `lhs_batching_dims` and `lhs_contracting_dims` are [`DotDimensionsAttributeRef::lhs_batching_dimensions`]
///     and [`DotDimensionsAttributeRef::lhs_contracting_dimensions`] in [`DotGeneralOperation::dimensions`].
///   - `rhs_result_dims = [d for d in axes(rhs) and d not in rhs_batching_dims and d not in rhs_contracting_dims]`,
///     where `rhs_batching_dims` and `rhs_contracting_dims` are [`DotDimensionsAttributeRef::rhs_batching_dimensions`]
///     and [`DotDimensionsAttributeRef::rhs_contracting_dimensions`] in [`DotGeneralOperation::dimensions`].
///   - `result_batching_index + result_lhs_index + result_rhs_index = result_index`, where
///     `size(result_batching_index) = size(lhs_batching_dims)`, `size(result_lhs_index) = size(lhs_result_dims)`,
///     and `size(result_rhs_index) = size(rhs_result_dims)`,
///   - `transposed_lhs = transpose(lhs, lhs_batching_dims + lhs_result_dims + lhs_contracting_dims)`,
///   - `transposed_lhs_slice = slice(transposed_lhs, result_batching_index + result_lhs_index + [:, ..., :])`,
///   - `reshaped_lhs_slice = reshape(transposed_lhs_slice, dims(lhs, lhs_contracting_dims))`,
///   - `transposed_rhs = transpose(rhs, rhs_batching_dims + rhs_result_dims + rhs_contracting_dims)`,
///   - `transposed_rhs_slice = slice(transposed_rhs, result_batching_index + result_rhs_index + [:, ..., :])`,
///   - `reshaped_rhs_slice = reshape(transposed_rhs_slice, dims(rhs, rhs_contracting_dims))`, and
///   - ```text
///     dot_product = reduce(
///         inputs=[multiply(reshaped_lhs_slice, reshaped_rhs_slice)],
///         initial_values=[constant(0, element_type(result))],
///         dimensions=range(size(lhs_contracting_dims)),
///         body=lambda x, y: add(x, y),
///     )
///     ```
///
/// For quantized types, this operation first dequantizes the input, applies the dot general operation
/// as described above, and then quantizes the result.
///
/// Furthermore:
///
///   - [`DotGeneralOperation::precision`] controls the tradeoff between speed and accuracy for computations
///     on different accelerator backends. It contains two [`Precision`]s; one that corresponds to the left-hand side
///     input and one that corresponds to the right-hand side input. The possible values for each [`Precision`] value
///     have the following semantics:
/// 
///       - [`Precision::Default`]: Fastest calculation, but least accurate approximation to the original number.
///       - [`Precision::High`]: Slower calculation, but more accurate approximation to the original number.
///       - [`Precision::Highest`]: Slowest calculation, but most accurate approximation to the original number.
/// 
///   - [`DotGeneralOperation::algorithm`] defines the main properties of the algorithm used to implement the dot
///     operation, which also defines the precision to use. Therefore, if the precision-related fields of the algorithm
///     are set, [`DotGeneralOperation::precision`] must not also be set. The fields of a
///     [`DotAlgorithmAttributeRef`] have the following semantics:
/// 
///       - [`DotAlgorithmAttributeRef::lhs_precision_type`] and [`DotAlgorithmAttributeRef::rhs_precision_type`] are
///         the precision types that the left-hand side input and right-hand side input of the operation are rounded to.
///         Precision types are independent of the storage types of the inputs and the output.
///       - [`DotAlgorithmAttributeRef::accumulation_type`] is the precision type used for accumulation of values during
///         the computation for the dot general operation.
///       - [`DotAlgorithmAttributeRef::lhs_component_count`], [`DotAlgorithmAttributeRef::rhs_component_count`], and
///         [`DotAlgorithmAttributeRef::primitive_operation_count`] apply when we are using an algorithm which
///         decomposes the left-hand side input and/or the right-hand side input into multiple components and performs
///         multiple "primitive" dot operations on those values, usually to emulate a higher precision (e.g., the
///         approach described in [this paper](https://arxiv.org/abs/1904.06376)). For algorithms with no decomposition,
///         these values should be set to `1`.
///       - [`DotAlgorithmAttributeRef::allow_imprecise_accumulation`] specifies whether accumulation in lower precision
///         is permitted for some of the computation steps (e.g., using `CUBLASLT_MATMUL_DESC_FAST_ACCUM`).
/// 
///     The following are some example supported [`DotAlgorithmAttributeRef`] values rendered with their MLIR rendering:
/// 
///     ```mlir
///     // Inputs are casted to `tf32`, and then accumulated in `f32`:
///     {lhs_precision_type = tf32,
///      rhs_precision_type = tf32,
///      accumulation_type = f32,
///      lhs_component_count = 1,
///      rhs_component_count = 1,
///      num_primitive_operations = 1,
///      allow_imprecise_accumulation = false}
///
///     // `bf16_6x`: each input is decomposed to 3 `bf16` components,
///     // then 6 dot operations are performed on those components,
///     // and the result is accumulated in `f32`.
///     {lhs_precision_type = bf16,
///      rhs_precision_type = bf16,
///      accumulation_type = f32,
///      lhs_component_count = 3,
///      rhs_component_count = 3,
///      num_primitive_operations = 6,
///      allow_imprecise_accumulation = false}
///
///     // Inputs are casted to `f8e5m2` and we accumulate in `f32`, but for some steps
///     // we may accumulate in lower precision.
///     {lhs_precision_type = f8e5m2,
///      rhs_precision_type = f8e5m2,
///      accumulation_type = f32,
///      lhs_component_count = 1,
///      rhs_component_count = 1,
///      num_primitive_operations = 1,
///      allow_imprecise_accumulation = true}
///     ```
/// 
///     It is up to the compiler implementation to decide which combinations are supported. In general, it is not
///     guaranteed that each algorithm is supported on each accelerator type by the consumer of StableHLO. If a given
///     algorithm is not supported, an error should be raised as opposed to falling back to an alternative.
///     StableHLO verification will provide a best effort verification, preventing algorithms that are not
///     known to be supported on any hardware.
///
///     You can refer [here](https://github.com/openxla/xla/blob/65aedb868f44ed75446b0fd770f848b7c9751293/xla/xla_data.proto#L1267)
///     for a list of known and likely supported widely algorithms.
///
/// # Example
///
/// The following is an example of a [`DotGeneralOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [
/// //        [[1, 2],
/// //         [3, 4]],
/// //        [[5, 6],
/// //         [7, 8]]
/// //       ]
/// // %rhs: [
/// //        [[1, 0],
/// //         [0, 1]],
/// //        [[1, 0],
/// //         [0, 1]]
/// //       ]
/// %result = stablehlo.dot_general %lhs, %rhs,
///   batching_dims = [0] x [0],
///   contracting_dims = [2] x [1],
///   algorithm = <
///     lhs_precision_type = tf32,
///     rhs_precision_type = tf32,
///     accumulation_type = f32,
///     lhs_component_count = 1,
///     rhs_component_count = 1,
///     num_primitive_operations = 1,
///     allow_imprecise_accumulation = false
///   > : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
/// // %result: [
/// //           [[1, 2],
/// //            [3, 4]],
/// //           [[5, 6],
/// //            [7, 8]]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#dot_general)
/// for more information.
pub trait DotGeneralOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand side input of this [`DotGeneralOperation`].
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the right-hand side input of this [`DotGeneralOperation`].
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns the [`DotDimensionsAttributeRef`] of this [`DotGeneralOperation`], specifying its
    /// batching and contracting dimensions.
    fn dimensions(&self) -> DotDimensionsAttributeRef<'c, 't> {
        self.attribute(DOT_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DotDimensionsAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{DOT_DIMENSIONS_ATTRIBUTE}' attribute in `stable_hlo::dot_general`"))
    }

    /// Returns the [`Precision`] configuration of this [`DotGeneralOperation`], if specified. This configuration
    /// consists of two [`Precision`]s; one that corresponds to the left-hand side input and one that corresponds
    /// to the right-hand side input.
    fn precision(&self) -> Option<(Precision, Precision)> {
        let error_message = format!("invalid '{DOT_PRECISION_ATTRIBUTE}' attribute in `stable_hlo::dot_general`");
        self.attribute(DOT_PRECISION_ATTRIBUTE).and_then(|attribute| {
            attribute.cast::<ArrayAttributeRef>().map(|attribute| {
                let mut elements = attribute
                    .elements()
                    .flat_map(|element| element.cast::<PrecisionAttributeRef>().map(|attribute| attribute.value()));
                (elements.next().expect(&error_message), elements.next().expect(&error_message))
            })
        })
    }

    /// Returns the [`DotAlgorithmAttributeRef`] of this [`DotGeneralOperation`], if specified.
    fn algorithm(&self) -> Option<DotAlgorithmAttributeRef<'c, 't>> {
        self.attribute(DOT_ALGORITHM_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DotAlgorithmAttributeRef>())
    }
}

mlir_op!(DotGeneral);
mlir_op_trait!(DotGeneral, OneResult);
mlir_op_trait!(DotGeneral, ZeroRegions);
mlir_op_trait!(DotGeneral, ZeroSuccessors);

/// Constructs a new detached/owned [`DotGeneralOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`DotGeneralOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn dot_general<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    dimensions: DotDimensionsAttributeRef<'c, 't>,
    precision: Option<(Precision, Precision)>,
    algorithm: Option<DotAlgorithmAttributeRef<'c, 't>>,
    result_type: T,
    location: L,
) -> DetachedDotGeneralOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let mut builder = OperationBuilder::new("stablehlo.dot_general", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .add_attribute(DOT_DIMENSIONS_ATTRIBUTE, dimensions);
    if let Some((lhs_precision, rhs_precision)) = precision {
        builder = builder.add_attribute(
            DOT_PRECISION_ATTRIBUTE,
            context.array_attribute(&[
                context.stable_hlo_precision(lhs_precision),
                context.stable_hlo_precision(rhs_precision),
            ]),
        );
    }
    if let Some(algorithm) = algorithm {
        builder = builder.add_attribute(DOT_ALGORITHM_ATTRIBUTE, algorithm);
    }
    builder
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::dot_general`")
}

/// StableHLO [`Attribute`] that models the dimension information in [`ConvolutionOperation`]s.
#[derive(Copy, Clone)]
pub struct ConvolutionDimensionsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> ConvolutionDimensionsAttributeRef<'c, 't> {
    mlir_attribute_field!(
        input_batch_dimension,
        ConvDimensionNumbersGetInputBatchDimension,
        usize,
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        input_feature_dimension,
        ConvDimensionNumbersGetInputFeatureDimension,
        usize,
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        input_spatial_dimensions,
        ConvDimensionNumbersGetInputSpatialDimensions,
        [usize],
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        kernel_input_feature_dimension,
        ConvDimensionNumbersGetKernelInputFeatureDimension,
        usize,
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        kernel_output_feature_dimension,
        ConvDimensionNumbersGetKernelOutputFeatureDimension,
        usize,
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        kernel_spatial_dimensions,
        ConvDimensionNumbersGetKernelSpatialDimensions,
        [usize],
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        output_batch_dimension,
        ConvDimensionNumbersGetOutputBatchDimension,
        usize,
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        output_feature_dimension,
        ConvDimensionNumbersGetOutputFeatureDimension,
        usize,
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(
        output_spatial_dimensions,
        ConvDimensionNumbersGetOutputSpatialDimensions,
        [usize],
        mlir_prefix = stablehlo,
    );
}

mlir_subtype_trait_impls!(
    ConvolutionDimensionsAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = ConvDimensionNumbers,
    mlir_prefix = stablehlo,
);

impl<'t> Context<'t> {
    /// Creates a new StableHLO [`ConvolutionDimensionsAttributeRef`] owned by this [`Context`].
    #[allow(clippy::too_many_arguments)]
    pub fn stable_hlo_convolution_dimensions<'c>(
        &'c self,
        input_batch_dimension: usize,
        input_feature_dimension: usize,
        input_spatial_dimensions: &[usize],
        kernel_input_feature_dimension: usize,
        kernel_output_feature_dimension: usize,
        kernel_spatial_dimensions: &[usize],
        output_batch_dimension: usize,
        output_feature_dimension: usize,
        output_spatial_dimensions: &[usize],
    ) -> ConvolutionDimensionsAttributeRef<'c, 't> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo());
        let input_spatial_dimensions = input_spatial_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let kernel_spatial_dimensions = kernel_spatial_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let output_spatial_dimensions = output_spatial_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>();
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            ConvolutionDimensionsAttributeRef::from_c_api(
                stablehloConvDimensionNumbersGet(
                    *self.handle.borrow(),
                    input_batch_dimension as i64,
                    input_feature_dimension as i64,
                    input_spatial_dimensions.len().cast_signed(),
                    input_spatial_dimensions.as_ptr(),
                    kernel_input_feature_dimension as i64,
                    kernel_output_feature_dimension as i64,
                    kernel_spatial_dimensions.len().cast_signed(),
                    kernel_spatial_dimensions.as_ptr(),
                    output_batch_dimension as i64,
                    output_feature_dimension as i64,
                    output_spatial_dimensions.len().cast_signed(),
                    output_spatial_dimensions.as_ptr(),
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Name of the [`Attribute`] that is used to store [`StaticOrDynamicConvolutionOperation::dimensions`].
pub const CONVOLUTION_DIMENSIONS_ATTRIBUTE: &str = "dimension_numbers";

/// Name of the [`Attribute`] that is used to store [`StaticOrDynamicConvolutionOperation::batch_group_count`].
pub const CONVOLUTION_BATCH_GROUP_COUNT_ATTRIBUTE: &str = "batch_group_count";

/// Name of the [`Attribute`] that is used to store [`StaticOrDynamicConvolutionOperation::feature_group_count`].
pub const CONVOLUTION_FEATURE_GROUP_COUNT_ATTRIBUTE: &str = "feature_group_count";

/// Name of the [`Attribute`] that is used to store [`StaticOrDynamicConvolutionOperation::window_strides`].
pub const CONVOLUTION_WINDOW_STRIDES_ATTRIBUTE: &str = "window_strides";

/// Name of the [`Attribute`] that is used to store [`StaticOrDynamicConvolutionOperation::lhs_dilation`].
pub const CONVOLUTION_LHS_DILATION_ATTRIBUTE: &str = "lhs_dilation";

/// Name of the [`Attribute`] that is used to store [`StaticOrDynamicConvolutionOperation::rhs_dilation`].
pub const CONVOLUTION_RHS_DILATION_ATTRIBUTE: &str = "rhs_dilation";

/// Name of the [`Attribute`] that is used to store [`StaticOrDynamicConvolutionOperation::window_reversal`].
pub const CONVOLUTION_WINDOW_REVERSAL_ATTRIBUTE: &str = "window_reversal";

/// Name of the [`Attribute`] that is used to store [`StaticOrDynamicConvolutionOperation::precision`].
pub const CONVOLUTION_PRECISION_ATTRIBUTE: &str = "precision_config";

/// Trait that is shared by [`ConvolutionOperation`] and [`DynamicConvolutionOperation`].
pub trait StaticOrDynamicConvolutionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input (i.e., its first operand) of this [`StaticOrDynamicConvolutionOperation`].
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the kernel (i.e., its second operand) of this [`StaticOrDynamicConvolutionOperation`].
    fn kernel(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns the [`ConvolutionDimensionsAttributeRef`] of this [`Operation`].
    fn dimensions(&self) -> ConvolutionDimensionsAttributeRef<'c, 't> {
        self.attribute(CONVOLUTION_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ConvolutionDimensionsAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{CONVOLUTION_DIMENSIONS_ATTRIBUTE}' attribute in `stable_hlo::convolution` or `stable_hlo::dynamic_conv`"))
    }

    /// Returns the batch group count of this [`Operation`].
    fn batch_group_count(&self) -> usize {
        self.attribute(CONVOLUTION_BATCH_GROUP_COUNT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as usize)
            .unwrap_or_else(|| panic!("invalid '{CONVOLUTION_BATCH_GROUP_COUNT_ATTRIBUTE}' attribute in `stable_hlo::convolution` or `stable_hlo::dynamic_conv`"))
    }

    /// Returns the feature group count of this [`Operation`].
    fn feature_group_count(&self) -> usize {
        self.attribute(CONVOLUTION_FEATURE_GROUP_COUNT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as usize)
            .unwrap_or_else(|| panic!("invalid '{CONVOLUTION_FEATURE_GROUP_COUNT_ATTRIBUTE}' attribute in `stable_hlo::convolution` or `stable_hlo::dynamic_conv`"))
    }

    /// Returns the window strides of this [`Operation`], if specified. The window strides specify how large
    /// of a jump we take each time with the sliding window for each _spatial_ (i.e., non-batch-or-feature) dimension of
    /// the left-hand side input. All stride values default to one when not specified.
    fn window_strides(&self) -> Option<Vec<usize>> {
        self.attribute(CONVOLUTION_WINDOW_STRIDES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
    }

    /// Returns the left-hand side input dilation of this [`Operation`], if specified. The dilation values
    /// specify how much we _dilate_ each _spatial_ (i.e., non-batch-or-feature) dimension of the left-hand side input.
    /// The dilation value for a specific dimension is interpreted as the spacing of the left-hand side input values
    /// along that dimension that each consecutive value of the kernel/filter interacts with. All dilation values
    /// default to one when not specified.
    ///
    /// Refer to this [blog post](https://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/)
    /// for more information on dilated convolutions.
    fn lhs_dilation(&self) -> Option<Vec<usize>> {
        self.attribute(CONVOLUTION_LHS_DILATION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
    }

    /// Returns the right-hand side input dilation of this [`Operation`], if specified. The dilation values
    /// specify how much we _dilate_ each _spatial_ (i.e., non-batch-or-feature) dimension of the right-hand side input.
    /// The dilation value for a specific dimension is interpreted as the spacing of the right-hand side input values
    /// along that dimension that each consecutive value of the kernel/filter interacts with. All dilation values
    /// default to one when not specified.
    ///
    /// Refer to this [blog post](https://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/)
    /// for more information on dilated convolutions.
    fn rhs_dilation(&self) -> Option<Vec<usize>> {
        self.attribute(CONVOLUTION_RHS_DILATION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
    }

    /// Returns an optional [`Vec`] that specifies, for each _spatial_ (i.e., non-batch-or-feature) dimension of the
    /// convolution window, if that dimension of the window should be [reversed](crate::dialects::stable_hlo::reverse)
    /// or not. Defaults to `false` for all spatial dimensions, if not specified.
    fn window_reversal(&self) -> Option<Vec<bool>> {
        self.attribute(CONVOLUTION_WINDOW_REVERSAL_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseBooleanArrayAttributeRef>())
            .map(|attribute| attribute.values().collect())
    }

    /// Returns the [`Precision`] configuration of this [`Operation`], if specified. That configuration
    /// consists of two [`Precision`]s; one that corresponds to the left-hand side input and one that corresponds
    /// to the right-hand side input.
    fn precision(&self) -> Option<(Precision, Precision)> {
        let error_message = format!(
            "invalid '{CONVOLUTION_PRECISION_ATTRIBUTE}' attribute in `stable_hlo::convolution` or `stable_hlo::dynamic_conv`",
        );
        self.attribute(CONVOLUTION_PRECISION_ATTRIBUTE).and_then(|attribute| {
            attribute.cast::<ArrayAttributeRef>().map(|attribute| {
                let mut elements = attribute
                    .elements()
                    .flat_map(|element| element.cast::<PrecisionAttributeRef>().map(|attribute| attribute.value()));
                (elements.next().expect(&error_message), elements.next().expect(&error_message))
            })
        })
    }
}

/// StableHLO [`Operation`] that computes convolutions over tensors (i.e., dot products between windows of one tensor
/// and slices of another).
///
/// More formally, consider the following reframing of the inputs in terms of `lhs` (i.e., the first input/operand) of
/// this operation, in order to be able to express windows of `lhs`:
///
/// ```text
/// lhs_window_dimensions = lhs_shape(
///   dim(lhs, ConvolutionOperation::dimensions().input_batch_dimension()),
///   dim(rhs, ConvolutionOperation::dimensions().kernel_spatial_dimensions()),
///   dim(lhs, ConvolutionOperation::dimensions().input_feature_dimension()),
/// )
///
/// lhs_window_strides = lhs_shape(1, ConvolutionOperation::window_strides(), 1)
///
/// lhs_padding = lhs_shape([0, 0], ConvolutionOperation::padding(), [0, 0])
///
/// lhs_base_dilations = lhs_shape(1, ConvolutionOperation::lhs_dilation(), 1)
///
/// lhs_window_dilations = lhs_shape(1, ConvolutionOperation::rhs_dilation(), 1)
/// ```
///
/// where:
///
/// ```text
/// lhs_shape(n, hw, c) = permute(
///   [n] + hw + [c],
///   [ConvolutionOperation::dimensions().input_batch_dimension()]
///   + ConvolutionOperation::dimensions().input_spatial_dimensions()
///   + [ConvolutionOperation::dimensions().input_feature_dimension()],
/// )
///
/// result_shape(n1, hw, c1) = permute(
///   [n1] + hw + [c1],
///   [ConvolutionOperation::dimensions().output_batch_dimension()]
///   + ConvolutionOperation::dimensions().output_spatial_dimensions()
///   + [ConvolutionOperation::dimensions().output_feature_dimension()],
/// )
///
/// permute([i[permutation[0]], i[permutation[1]], ..., i[permutation[R-1]]], permutation) = [i[0], i[1], ..., i[R-1]]
/// ```
///
/// If `ConvolutionOperation::feature_group_count() = 1` and `ConvolutionOperation::batch_group_count() = 1`, then for
/// all `output_spatial_index` values in
/// `index_space(dim(result, ConvolutionOperation::dimensions().output_spatial_dimensions()...))`,
/// `result[result_shape(:, output_spatial_index, :)] = dot_product` where:
///
/// ```text
/// padding_value = constant(0, element_type(lhs))
///
/// padded_lhs = pad(lhs, padding_value, lhs_padding[:, 0], lhs_padding[:, 1], lhs_base_dilations - 1)
///
/// lhs_window_start = lhs_shape(0, output_spatial_index, 0) * lhs_window_strides
///
/// lhs_window = slice(padded_lhs, lhs_window_start, lhs_window_start + lhs_window_dimensions, lhs_window_dilations)
///
/// reversed_lhs_window = reverse(
///   lhs_window,
///   [
///     ConvolutionOperation::dimensions().input_spatial_dimensions()[dim]
///     for dim in range(size(ConvolutionOperation::window_reversal()))
///     if ConvolutionOperation::window_reversal()[dim] = true
///   ],
/// )
///
/// dot_product = dot_general(
///   reversed_lhs_window,
///   rhs,
///   lhs_batching_dimensions=[],
///   lhs_contracting_dimensions=(
///     ConvolutionOperation::dimensions().input_spatial_dimensions()
///     + [ConvolutionOperation::dimensions().input_feature_dimension()]
///   ),
///   rhs_batching_dimensions=[],
///   rhs_contracting_dimensions=(
///     ConvolutionOperation::dimensions().kernel_spatial_dimensions()
///     + [ConvolutionOperation::dimensions().kernel_input_feature_dimension()]
///   ),
/// )
/// ```
///
/// If `ConvolutionOperation::feature_group_count() > 1`:
///
/// ```text
/// lhses = split(
///   lhs,
///   ConvolutionOperation::feature_group_count(),
///   ConvolutionOperation::dimensions().input_feature_dimension(),
/// )
///
/// rhses = split(
///   rhs,
///   ConvolutionOperation::feature_group_count(),
///   ConvolutionOperation::dimensions().kernel_output_feature_dimension(),
/// )
///
/// results... = convolution(lhses..., rhses..., ..., feature_group_count=1, ...)
///
/// result = concatenate(results, ConvolutionOperation::dimensions().output_feature_dimension())
/// ```
///
/// If `ConvolutionOperation::batch_group_count() > 1`:
///
/// ```text
/// lhses = split(
///   lhs,
///   ConvolutionOperation::batch_group_count(),
///   ConvolutionOperation::dimensions().input_batch_dimension(),
/// )
///
/// rhses = split(
///   rhs,
///   ConvolutionOperation::batch_group_count(),
///   ConvolutionOperation::dimensions().kernel_output_feature_dimension(),
/// )
///
/// results... = convolution(lhses..., rhses..., ..., batch_group_count=1, ...)
///
/// result = concatenate(results, ConvolutionOperation::dimensions().output_feature_dimension())
/// ```
///
/// For quantized types, this operation first dequantizes the input, applies the dot general operation
/// as described above, and then quantizes the result.
///
/// # Example
///
/// The following is an example of a [`ConvolutionOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %lhs: [[
/// //        [
/// //          [1], [2], [5], [6]
/// //        ],
/// //        [
/// //          [3], [4], [7], [8]
/// //        ],
/// //        [
/// //          [10], [11], [14], [15]
/// //        ],
/// //        [
/// //          [12], [13], [16], [17]
/// //        ]
/// //      ]]
/// // %rhs: [
/// //        [[[1]], [[1]], [[1]]],
/// //        [[[1]], [[1]], [[1]]],
/// //        [[[1]], [[1]], [[1]]]
/// //       ]
/// %result = stablehlo.convolution(%lhs, %rhs)
///   dim_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
///   window = {
///     stride = [4, 4],
///     pad = [[0, 0], [0, 0]],
///     lhs_dilate = [2, 2],
///     rhs_dilate = [1, 1],
///     reverse = [false, false]
///   } {
///     // In the StableHLO dialect, dimension numbers are encoded via:
///     // `[<input dimensions>]x[<kernel dimensions>]->[output dimensions]`.
///     // "b" is batch dimension, "f" is feature dimension,
///     // "i" is input feature dimension, "o" is output feature dimension,
///     // "0/1/etc" are spatial dimensions.
///     batch_group_count = 1 : i64,
///     feature_group_count = 1 : i64,
///     precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
///   } : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>) -> tensor<1x2x2x1xi64>
/// // %result: [[
/// //            [[10], [26]],
/// //            [[46], [62]]
/// //          ]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#convolution)
/// for more information.
pub trait ConvolutionOperation<'o, 'c: 'o, 't: 'c>:
    StaticOrDynamicConvolutionOperation<'o, 'c, 't> + HasPadding<'o, 'c, 't>
{
}

mlir_op!(Convolution);
mlir_op_trait!(Convolution, OneResult);
mlir_op_trait!(Convolution, ZeroRegions);
mlir_op_trait!(Convolution, ZeroSuccessors);
mlir_op_trait!(Convolution, @local HasPadding);
mlir_op_trait!(Convolution, @local StaticOrDynamicConvolutionOperation);

/// Constructs a new detached/owned [`ConvolutionOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ConvolutionOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
#[allow(clippy::too_many_arguments)]
pub fn convolution<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    dimensions: ConvolutionDimensionsAttributeRef<'c, 't>,
    batch_group_count: usize,
    feature_group_count: usize,
    window_strides: Option<&[usize]>,
    padding: Option<&[(usize, usize)]>,
    lhs_dilation: Option<&[usize]>,
    rhs_dilation: Option<&[usize]>,
    window_reversal: Option<&[bool]>,
    precision: Option<(Precision, Precision)>,
    result_type: T,
    location: L,
) -> DetachedConvolutionOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let i64_type = context.signless_integer_type(64);
    let mut builder = OperationBuilder::new("stablehlo.convolution", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .add_attribute(CONVOLUTION_DIMENSIONS_ATTRIBUTE, dimensions)
        .add_attribute(
            CONVOLUTION_BATCH_GROUP_COUNT_ATTRIBUTE,
            context.integer_attribute(i64_type, batch_group_count as i64),
        )
        .add_attribute(
            CONVOLUTION_FEATURE_GROUP_COUNT_ATTRIBUTE,
            context.integer_attribute(i64_type, feature_group_count as i64),
        );
    if let Some(window_strides) = window_strides {
        let window_strides = window_strides.iter().map(|v| *v as i64).collect::<Vec<_>>();
        builder = builder.add_attribute(
            CONVOLUTION_WINDOW_STRIDES_ATTRIBUTE,
            context.dense_i64_array_attribute(window_strides.as_slice()).unwrap(),
        );
    }
    if let Some(padding) = padding {
        builder = builder.add_attribute(PADDING_ATTRIBUTE, context.stable_hlo_padding(padding, location));
    }
    if let Some(lhs_dilation) = lhs_dilation {
        let lhs_dilation = lhs_dilation.iter().map(|v| *v as i64).collect::<Vec<_>>();
        builder = builder.add_attribute(
            CONVOLUTION_LHS_DILATION_ATTRIBUTE,
            context.dense_i64_array_attribute(lhs_dilation.as_slice()).unwrap(),
        );
    }
    if let Some(rhs_dilation) = rhs_dilation {
        let rhs_dilation = rhs_dilation.iter().map(|v| *v as i64).collect::<Vec<_>>();
        builder = builder.add_attribute(
            CONVOLUTION_RHS_DILATION_ATTRIBUTE,
            context.dense_i64_array_attribute(rhs_dilation.as_slice()).unwrap(),
        );
    }
    if let Some(window_reversal) = window_reversal {
        builder = builder.add_attribute(
            CONVOLUTION_WINDOW_REVERSAL_ATTRIBUTE,
            context.dense_bool_array_attribute(window_reversal).unwrap(),
        );
    }
    if let Some((lhs_precision, rhs_precision)) = precision {
        builder = builder.add_attribute(
            CONVOLUTION_PRECISION_ATTRIBUTE,
            context.array_attribute(&[
                context.stable_hlo_precision(lhs_precision),
                context.stable_hlo_precision(rhs_precision),
            ]),
        );
    }
    builder
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::convolution`")
}

/// StableHLO [`Operation`] that computes convolutions over tensors (i.e., dot products between windows of one tensor
/// and slices of another). This operation is a dynamic variant of [`ConvolutionOperation`] which has the exact same
/// semantics except for the fact that instead of a static padding attribute, the padding is dynamic and provided as
/// the third input/operand of this operation.
///
/// # Example
///
/// ```mlir
/// // %lhs: [[
/// //        [[1], [2], [5], [6]],
/// //        [[3], [4], [7], [8]],
/// //        [[10], [11], [14], [15]],
/// //        [[12], [13], [16], [17]]
/// //      ]]
/// //
/// // %rhs: [
/// //         [[[1]], [[1]], [[1]]],
/// //         [[[1]], [[1]], [[1]]],
/// //         [[[1]], [[1]], [[1]]]
/// //        ]
/// // %padding: [[1, 1],
/// //            [1, 1]]
/// %result = "stablehlo.dynamic_conv"(%lhs, %rhs, %padding) <{
///   batch_group_count = 1 : i64,
///   dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
///   feature_group_count = 1 : i64,
///   lhs_dilation = array<i64: 2, 2>,
///   precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
///   rhs_dilation = array<i64: 1, 1>,
///   window_reversal = array<i1: false, false>,
///   window_strides = array<i64: 4, 4>
/// }> : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>, tensor<2x2xi64>) -> tensor<1x2x2x1xi64>
/// // %result: [[
/// //            [[1], [5]],
/// //            [[10], [14]]
/// //          ]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#dynamic_conv)
/// for more information.
pub trait DynamicConvolutionOperation<'o, 'c: 'o, 't: 'c>: StaticOrDynamicConvolutionOperation<'o, 'c, 't> {
    /// Returns the padding of this [`DynamicConvolutionOperation`]. The padding consists of a pair of numbers
    /// for each _spatial_ (i.e., non-batch-or-feature) dimension of the left-hand side input and is represented as a
    /// two-dimensional tensor with shape `[S, 2]` where `S` is the number of _spatial_ dimensions. The first number
    /// in each pair specifies the amount of padding inserted _before_ the values of the tensor on that dimension and
    /// the second number specifies the amount of padding inserted _after_ the values of the tensor on that dimension.
    fn padding(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(2).unwrap()
    }
}

mlir_op!(DynamicConvolution);
mlir_op_trait!(DynamicConvolution, OneResult);
mlir_op_trait!(DynamicConvolution, ZeroRegions);
mlir_op_trait!(DynamicConvolution, ZeroSuccessors);
mlir_op_trait!(DynamicConvolution, @local StaticOrDynamicConvolutionOperation);

/// Constructs a new detached/owned [`DynamicConvolutionOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`DynamicConvolutionOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
#[allow(clippy::too_many_arguments)]
pub fn dynamic_convolution<
    'lhs,
    'rhs,
    'p,
    'c: 'lhs + 'rhs + 'p,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'rhs, 'c, 't>,
    P: Value<'p, 'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    padding: P,
    dimensions: ConvolutionDimensionsAttributeRef<'c, 't>,
    batch_group_count: usize,
    feature_group_count: usize,
    window_strides: Option<&[usize]>,
    lhs_dilation: Option<&[usize]>,
    rhs_dilation: Option<&[usize]>,
    window_reversal: Option<&[bool]>,
    precision: Option<(Precision, Precision)>,
    result_type: T,
    location: L,
) -> DetachedDynamicConvolutionOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let i64_type = location.context().signless_integer_type(64);
    let mut builder = OperationBuilder::new("stablehlo.dynamic_conv", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .add_operand(padding)
        .add_attribute(CONVOLUTION_DIMENSIONS_ATTRIBUTE, dimensions)
        .add_attribute(
            CONVOLUTION_BATCH_GROUP_COUNT_ATTRIBUTE,
            context.integer_attribute(i64_type, batch_group_count as i64),
        )
        .add_attribute(
            CONVOLUTION_FEATURE_GROUP_COUNT_ATTRIBUTE,
            context.integer_attribute(i64_type, feature_group_count as i64),
        );
    if let Some(window_strides) = window_strides {
        let window_strides = window_strides.iter().map(|v| *v as i64).collect::<Vec<_>>();
        builder = builder.add_attribute(
            CONVOLUTION_WINDOW_STRIDES_ATTRIBUTE,
            context.dense_i64_array_attribute(window_strides.as_slice()).unwrap(),
        );
    }
    if let Some(lhs_dilation) = lhs_dilation {
        let lhs_dilation = lhs_dilation.iter().map(|v| *v as i64).collect::<Vec<_>>();
        builder = builder.add_attribute(
            CONVOLUTION_LHS_DILATION_ATTRIBUTE,
            context.dense_i64_array_attribute(lhs_dilation.as_slice()).unwrap(),
        );
    }
    if let Some(rhs_dilation) = rhs_dilation {
        let rhs_dilation = rhs_dilation.iter().map(|v| *v as i64).collect::<Vec<_>>();
        builder = builder.add_attribute(
            CONVOLUTION_RHS_DILATION_ATTRIBUTE,
            context.dense_i64_array_attribute(rhs_dilation.as_slice()).unwrap(),
        );
    }
    if let Some(window_reversal) = window_reversal {
        builder = builder.add_attribute(
            CONVOLUTION_WINDOW_REVERSAL_ATTRIBUTE,
            context.dense_bool_array_attribute(window_reversal).unwrap(),
        );
    }
    if let Some((lhs_precision, rhs_precision)) = precision {
        builder = builder.add_attribute(
            CONVOLUTION_PRECISION_ATTRIBUTE,
            context.array_attribute(&[
                context.stable_hlo_precision(lhs_precision),
                context.stable_hlo_precision(rhs_precision),
            ]),
        );
    }
    builder
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::dynamic_convolution`")
}

/// Name of the [`Attribute`] that is used to store [`CholeskyOperation::lower`].
pub const CHOLESKY_LOWER_ATTRIBUTE: &str = "lower";

/// StableHLO [`Operation`] that computes the [Cholesky](https://en.wikipedia.org/wiki/Cholesky_decomposition)
/// decomposition of one or more square symmetric [positive-definite](https://en.wikipedia.org/wiki/Definite_matrix)
/// matrices. More formally, for all `i` in `index_space(result)`, `result[i[0], ..., i[R-3], :, :]` is a Cholesky
/// decomposition of `input[i[0], ..., i[R-3], :, :]` in the form of either of a lower-triangular matrix, if
/// [`CholeskyOperation::lower`] is `true`, or an upper-triangular matrix. The output values in the opposite triangle
/// (i.e., the strict upper or lower triangle, respectively) are implementation-specific. If there exists an `i` where
/// the input matrix is not a Hermitian positive-definite matrix, then the behavior of this operation is undefined.
///
/// For quantized types, this operation first dequantizes the input, applies the Cholesky operation as described above,
/// and then quantizes the result.
///
/// # Example
///
/// The following is an example of a [`CholeskyOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %a: [
/// //      [1.0, 2.0, 3.0],
/// //      [2.0, 20.0, 26.0],
/// //      [3.0, 26.0, 70.0]
/// //     ]
/// %result = stablehlo.cholesky %a, lower = true : tensor<3x3xf64>
/// // %result: [
/// //           [1.0, 0.0, 0.0],
/// //           [2.0, 4.0, 0.0],
/// //           [3.0, 5.0, 6.0]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#cholesky)
/// for more information.
pub trait CholeskyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns whether this [`CholeskyOperation`] computes lower or an upper triangular matrices as its result.
    fn lower(&self) -> bool {
        self.attribute(CHOLESKY_LOWER_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| panic!("invalid '{CHOLESKY_LOWER_ATTRIBUTE}' attribute in `stable_hlo::cholesky`"))
    }
}

mlir_op!(Cholesky);
mlir_op_trait!(Cholesky, OneResult);
mlir_op_trait!(Cholesky, ZeroRegions);
mlir_op_trait!(Cholesky, ZeroSuccessors);

/// Constructs a new detached/owned [`CholeskyOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`CholeskyOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn cholesky<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    lower: bool,
    location: L,
) -> DetachedCholeskyOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.cholesky", location)
        .add_operand(input)
        .add_attribute(CHOLESKY_LOWER_ATTRIBUTE, location.context().boolean_attribute(lower))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::cholesky`")
}

mlir_enum_attribute!(
    rust_name = TriangularSolveTransposeType,
    mlir_name = Transpose,
    description = "StableHLO [`TriangularSolveOperation`] transpose type",
    variants = {
        NoTranspose => "NO_TRANSPOSE",
        Transpose => "TRANSPOSE",
        Adjoint => "ADJOINT",
    },
    rust_prefix = stable_hlo,
    mlir_prefix = stablehlo,
    mlir_dialect_handle_constructor = stable_hlo,
);

/// Name of the [`Attribute`] that is used to store [`TriangularSolveOperation::left_side`].
pub const TRIANGULAR_SOLVE_LEFT_SIDE_ATTRIBUTE: &str = "left_side";

/// Name of the [`Attribute`] that is used to store [`TriangularSolveOperation::lower`].
pub const TRIANGULAR_SOLVE_LOWER_ATTRIBUTE: &str = "lower";

/// Name of the [`Attribute`] that is used to store [`TriangularSolveOperation::unit_diagonal`].
pub const TRIANGULAR_SOLVE_UNIT_DIAGONAL_ATTRIBUTE: &str = "unit_diagonal";

/// Name of the [`Attribute`] that is used to store [`TriangularSolveOperation::transpose_a`].
pub const TRIANGULAR_SOLVE_TRANSPOSE_A_ATTRIBUTE: &str = "transpose_a";

/// StableHLO [`Operation`] that solves triangular systems of linear equations with lower or upper triangular
/// coefficient matrices. More formally, given [`TriangularSolveOperation::a`] and [`TriangularSolveOperation::b`],
/// `result[i[0], ..., i[R-3], :, :]` is the solution to:
///
///   - `op(a[i[0], ..., i[R-3], :, :]) * x = b[i[0], ..., i[R-3], :, :]`, when [`TriangularSolveOperation::left_side`]
///     is `true`, and
///   - `x * op(a[i0, ..., iR-3, :, :]) = b[i0, ..., iR-3, :, :]`, otherwise.
///
/// solving for `x` where `op(a)` is determined by [`TriangularSolveOperation::transpose_a`]:
///
///   - [`TriangularSolveTransposeType::NoTranspose`]: `op` is the identity function.
///   - [`TriangularSolveTransposeType::Transpose`]: `op` computes the transpose of `a`.
///   - [`TriangularSolveTransposeType::Adjoint`]: `op` computes the conjugate transpose of `a`.
///
/// The input data is read only from the lower triangle of `a`, if [`TriangularSolveOperation::lower`] is `true`, and
/// the upper triangle of `a`, otherwise. The output data is returned in the same triangle; the values in the other
/// triangle are implementation-specific.
///
/// If [`TriangularSolveOperation::unit_diagonal`] is `true` then the implementation can assume that the diagonal
/// elements of [`TriangularSolveOperation::a`] are all equal to `1`. Violating this assumption results in undefined
/// behavior.
///
/// For quantized types, this operation first dequantizes the input, applies the triangular solve operation as described
/// above, and then quantizes the result.
///
/// # Example
///
/// The following is an example of a [`TriangularSolveOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %a = [
/// //       [1.0, 0.0, 0.0],
/// //       [2.0, 4.0, 0.0],
/// //       [3.0, 5.0, 6.0]
/// //      ]
/// // %b = [
/// //       [2.0, 0.0, 0.0],
/// //       [4.0, 8.0, 0.0],
/// //       [6.0, 10.0, 12.0]
/// //      ]
/// %result = "stablehlo.triangular_solve"(%a, %b) <{
///   left_side = true,
///   lower = true,
///   transpose_a = #stablehlo<transpose NO_TRANSPOSE>,
///   unit_diagonal = false
/// }> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
/// // %result: [
/// //           [2.0, 0.0, 0.0],
/// //           [0.0, 2.0, 0.0],
/// //           [0.0, 0.0, 2.0]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#triangular_solve)
/// for more information.
pub trait TriangularSolveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the first input of this [`TriangularSolveOperation`].
    fn a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the second input of this [`TriangularSolveOperation`].
    fn b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns whether the coefficient matrix is placed on the left side for this [`TriangularSolveOperation`].
    fn left_side(&self) -> bool {
        self.attribute(TRIANGULAR_SOLVE_LEFT_SIDE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| panic!("invalid '{TRIANGULAR_SOLVE_LEFT_SIDE_ATTRIBUTE}' attribute in `stable_hlo::triangular_solve`"))
    }

    /// Returns whether this [`TriangularSolveOperation`] operates on lower or upper triangular matrices.
    fn lower(&self) -> bool {
        self.attribute(TRIANGULAR_SOLVE_LOWER_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| panic!("invalid '{TRIANGULAR_SOLVE_LOWER_ATTRIBUTE}' attribute in `stable_hlo::triangular_solve`"))
    }

    /// Returns `true` if this [`TriangularSolveOperation`] can assume that the diagonal elements of
    /// [`TriangularSolveOperation::a`] are all equal to `1`
    fn unit_diagonal(&self) -> bool {
        self.attribute(TRIANGULAR_SOLVE_UNIT_DIAGONAL_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| panic!("invalid '{TRIANGULAR_SOLVE_UNIT_DIAGONAL_ATTRIBUTE}' attribute in `stable_hlo::triangular_solve`"))
    }

    /// Returns the [`TriangularSolveTransposeType`] for this [`TriangularSolveOperation`].
    fn transpose_a(&self) -> TriangularSolveTransposeType {
        self.attribute(TRIANGULAR_SOLVE_TRANSPOSE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TriangularSolveTransposeTypeAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| panic!("invalid '{TRIANGULAR_SOLVE_TRANSPOSE_A_ATTRIBUTE}' attribute in `stable_hlo::triangular_solve`"))
    }
}

mlir_op!(TriangularSolve);
mlir_op_trait!(TriangularSolve, OneResult);
mlir_op_trait!(TriangularSolve, ZeroRegions);
mlir_op_trait!(TriangularSolve, ZeroSuccessors);

/// Constructs a new detached/owned [`TriangularSolveOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`TriangularSolveOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn triangular_solve<
    'a,
    'b,
    'c: 'a + 'b,
    't: 'c,
    A: Value<'a, 'c, 't>,
    B: Value<'b, 'c, 't>,
    L: Location<'c, 't>,
>(
    a: A,
    b: B,
    left_side: bool,
    lower: bool,
    unit_diagonal: bool,
    transpose_a: TriangularSolveTransposeType,
    location: L,
) -> DetachedTriangularSolveOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.triangular_solve", location)
        .add_operand(a)
        .add_operand(b)
        .add_attribute(TRIANGULAR_SOLVE_LEFT_SIDE_ATTRIBUTE, context.boolean_attribute(left_side))
        .add_attribute(TRIANGULAR_SOLVE_LOWER_ATTRIBUTE, context.boolean_attribute(lower))
        .add_attribute(TRIANGULAR_SOLVE_UNIT_DIAGONAL_ATTRIBUTE, context.boolean_attribute(unit_diagonal))
        .add_attribute(
            TRIANGULAR_SOLVE_TRANSPOSE_A_ATTRIBUTE,
            context.stable_hlo_triangular_solve_transpose_type(transpose_a),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::triangular_solve`")
}

mlir_enum_attribute!(
    rust_name = FftType,
    mlir_name = FftType,
    description = "StableHLO [`FftOperation`] type",
    variants = {
        FFT => "FFT",
        IFFT => "IFFT",
        RFFT => "RFFT",
        IRFFT => "IRFFT",
    },
    rust_prefix = stable_hlo,
    mlir_prefix = stablehlo,
    mlir_dialect_handle_constructor = stable_hlo,
);

/// Name of the [`Attribute`] that is used to store [`FftOperation::fft_type`].
pub const FFT_TYPE_ATTRIBUTE: &str = "fft_type";

/// Name of the [`Attribute`] that is used to store [`FftOperation::fft_length`].
pub const FFT_LENGTH_ATTRIBUTE: &str = "fft_length";

/// StableHLO [`Operation`] that performs the forward and inverse Fast Fourier Transforms (FFTs) for real and complex
/// input and output tensors. The type of Fourier transform it performs is controlled by [`FftOperation::fft_type`]:
///
///   - [`FftType::FFT`]: Forward complex-to-complex FFT.
///   - [`FftType::IFFT`]: Inverse complex-to-complex FFT.
///   - [`FftType::RFFT`]: Forward real-to-complex FFT.
///   - [`FftType::IRFFT`]: Inverse real-to-complex FFT (i.e. the input is complex and the output real).
///
/// More formally, given the function `fft` which takes 1-dimensional tensors of complex types as input,
/// produces 1-dimensional tensors of the same types as output, and computes the discrete Fourier transform,
/// and `ifft` for its inverse, the result of [`FftOperation`] is defined as follows:
///
///   - For [`FftType::FFT`], the result is defined as the final result of a series of `L` computations where
///     `L` is the size of [`FftOperation::fft_length`]. For example, for `L = 3`:
///     - `result1[i[0], ..., :] = fft(operand[i[0], ..., :])`,
///     - `result2[i[0], ..., :, i[R-1]] = fft(result1[i[0], ..., :, i[R-1]])`, and
///     - `result[i[0], ..., :, i[R-2], i[R-1]] = fft(result2[i[0], ..., :, i[R-2], i[R-1]])`.
///   - For [`FftType::IFFT`], the result is defined as the inverse of the computations for [`FftType::FFT`].
///     For example, for `L = 3`:
///     - `result1[i[0], ..., :, i[R-2], i[R-1]] = ifft(operand[i[0], ..., :, i[R-2], i[R-1]])`,
///     - `result2[i[0], ..., :, i[R-1]] = ifft(result1[i[0], ..., :, i[R-1]])`, and
///     - `result[i[0], ..., :] = ifft(result2[i[0], ..., :])`.
///
/// Furthermore, given the function `rfft` which takes 1-dimensional tensors of floating-point types, produces
/// 1-dimensional tensors of complex types of the same floating-point semantics, and works as follows:
///
///   - `complex_operand... = (real_operand..., 0.0)`,
///   - `complex_result = fft(complex_operand)`, and
///   - `rfft(real_operand) = complex_result[:(rank(complex_result) / 2 + 1)]` (when the discrete Fourier transform is
///     computed for real operands, the first `N/2 + 1` elements of the result unambiguously define the rest of the
///     result, so the result of `rfft` is truncated to avoid computing redundant elements),
///
/// and its inverse `irfft`, the result of [`FftOperation`] is defined as follows:
///
///   - For [`FftType::RFFT`], the result is defined as the final result of a series of `L` computations where
///     `L` is the size of [`FftOperation::fft_length`]. For example, for `L = 3`:
///     - `result1[i[0], ..., :] = rfft(operand[i[0], ..., :])`,
///     - `result2[i[0], ..., :, i[R-1]] = fft(result1[i[0], ..., :, i[R-1]])`, and
///     - `result[i[0], ..., :, i[R-2], i[R-1]] = fft(result2[i[0], ..., :, i[R-2], i[R-1]])`.
///   - For [`FftType::IRFFT`], the result is defined as the inverse of the computations for [`FftType::RFFT`].
///     For example, for `L = 3`:
///     - `result1[i[0], ..., :, i[R-2], i[R-1]] = ifft(operand[i[0], ..., :, i[R-2], i[R-1]])`,
///     - `result2[i[0], ..., :, i[R-1]] = ifft(result1[i[0], ..., :, i[R-1]])`, and
///     - `result[i[0], ..., :] = irfft(result2[i[0], ..., :])`.
///
/// # Example
///
/// The following is an example of an [`FftOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
/// %result = stablehlo.fft %operand, type =  FFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
/// // %result: [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#fft)
/// for more information.
pub trait FftOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the [`FftType`] of this [`FftOperation`].
    fn fft_type(&self) -> FftType {
        self.attribute(FFT_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<FftTypeAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| panic!("invalid '{FFT_TYPE_ATTRIBUTE}' attribute in `stable_hlo::fft`"))
    }

    /// Returns the length (i.e., number of step-wise FFT transformations) of this [`FftOperation`].
    fn fft_length(&self) -> Vec<usize> {
        self.attribute(FFT_LENGTH_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
            .unwrap_or_else(|| panic!("invalid '{FFT_LENGTH_ATTRIBUTE}' attribute in `stable_hlo::fft`"))
    }
}

mlir_op!(Fft);
mlir_op_trait!(Fft, OneResult);
mlir_op_trait!(Fft, ZeroRegions);
mlir_op_trait!(Fft, ZeroSuccessors);

/// Constructs a new detached/owned [`FftOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`FftOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn fft<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    r#type: FftType,
    length: &[usize],
    location: L,
) -> DetachedFftOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.fft", location)
        .add_operand(input)
        .add_attribute(FFT_TYPE_ATTRIBUTE, context.stable_hlo_fft_type(r#type))
        .add_attribute(
            FFT_LENGTH_ATTRIBUTE,
            context
                .dense_i64_array_attribute(length.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::fft`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};
    use crate::dialects::func;
    use crate::{Attribute, Block, Context, Float8TypeRef, FloatTypeRef, Operation, Size, Type};

    use super::{
        CholeskyOperation, DotAlgorithmPreset, DotGeneralOperation, DynamicConvolutionOperation, FftOperation, FftType,
        HasPadding, Precision, StaticOrDynamicConvolutionOperation, TriangularSolveOperation,
        TriangularSolveTransposeType, cholesky, convolution, dot_general, dynamic_convolution, fft, triangular_solve,
    };

    #[test]
    fn test_precision_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_precision(Precision::Highest);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), Precision::Highest);
    }

    #[test]
    fn test_precision_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_precision(Precision::Highest);
        let attribute_2 = context.stable_hlo_precision(Precision::Highest);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_precision(Precision::High);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_precision(Precision::Highest);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_precision_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_precision(Precision::Highest);
        test_attribute_display_and_debug(attribute, "#stablehlo<precision HIGHEST>");
    }

    #[test]
    fn test_precision_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_precision(Precision::Highest);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_dot_dimensions_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_dot_dimensions(&[0, 1], &[2, 3], &[4], &[5]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.lhs_batching_dimensions(), vec![0, 1]);
        assert_eq!(attribute.rhs_batching_dimensions(), vec![2, 3]);
        assert_eq!(attribute.lhs_contracting_dimensions(), vec![4]);
        assert_eq!(attribute.rhs_contracting_dimensions(), vec![5]);
    }

    #[test]
    fn test_dot_dimensions_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_dot_dimensions(&[0, 1], &[2, 3], &[4], &[5]);
        let attribute_2 = context.stable_hlo_dot_dimensions(&[0, 1], &[2, 3], &[4], &[5]);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_dot_dimensions(&[2, 3], &[0], &[0], &[1]);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_dot_dimensions(&[0, 1], &[2, 3], &[4], &[5]);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_dot_dimensions_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_dot_dimensions(&[0, 1], &[2, 3], &[4], &[5]);
        test_attribute_display_and_debug(
            attribute,
            "#stablehlo.dot<\
              lhs_batching_dimensions = [0, 1], \
              rhs_batching_dimensions = [2, 3], \
              lhs_contracting_dimensions = [4], \
              rhs_contracting_dimensions = [5]\
            >",
        );
    }

    #[test]
    fn test_dot_dimensions_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_dot_dimensions(&[0, 1], &[2, 3], &[4], &[5]);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_dot_algorithm_attribute() {
        let context = Context::new();
        let f32_type = context.float32_type();
        let attribute = context.stable_hlo_dot_algorithm(f32_type, f32_type, f32_type, 2, 3, 4, true);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.lhs_precision_type(), f32_type);
        assert_eq!(attribute.rhs_precision_type(), f32_type);
        assert_eq!(attribute.accumulation_type(), f32_type);
        assert_eq!(attribute.lhs_component_count(), 2);
        assert_eq!(attribute.rhs_component_count(), 3);
        assert_eq!(attribute.primitive_operation_count(), 4);
        assert_eq!(attribute.allow_imprecise_accumulation(), true);
    }

    #[test]
    fn test_dot_algorithm_attribute_equality() {
        let context = Context::new();
        let f32_type = context.float32_type();
        let f64_type = context.float64_type();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_dot_algorithm(f32_type, f32_type, f32_type, 1, 1, 1, false);
        let attribute_2 = context.stable_hlo_dot_algorithm(f32_type, f32_type, f32_type, 1, 1, 1, false);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_dot_algorithm(f32_type, f64_type, f32_type, 1, 1, 2, true);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_dot_algorithm(f32_type, f32_type, f32_type, 1, 1, 1, false);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_dot_algorithm_attribute_display_and_debug() {
        let context = Context::new();
        let f32_type = context.float32_type();
        let f64_type = context.float64_type();
        let attribute = context.stable_hlo_dot_algorithm(f32_type, f64_type, f32_type, 4, 2, 1, false);
        test_attribute_display_and_debug(
            attribute,
            "#stablehlo.dot_algorithm<\
              lhs_precision_type = f32, \
              rhs_precision_type = f64, \
              accumulation_type = f32, \
              lhs_component_count = 4, \
              rhs_component_count = 2, \
              num_primitive_operations = 1, \
              allow_imprecise_accumulation = false\
            >",
        );
    }

    #[test]
    fn test_dot_algorithm_attribute_casting() {
        let context = Context::new();
        let f32_type = context.float32_type();
        let f64_type = context.float64_type();
        let attribute = context.stable_hlo_dot_algorithm(f32_type, f64_type, f32_type, 4, 2, 1, false);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_dot_algorithm_preset() {
        let context = Context::new();
        let f8e4m3fn_type = context.float8e4m3fn_type().as_ref().cast::<Float8TypeRef>().unwrap();
        let f8e5m2_type = context.float8e5m2_type().as_ref().cast::<Float8TypeRef>().unwrap();
        let f16_type = context.float16_type().as_ref().cast::<FloatTypeRef>().unwrap();
        let f32_type = context.float32_type().as_ref().cast::<FloatTypeRef>().unwrap();
        let f64_type = context.float64_type().as_ref().cast::<FloatTypeRef>().unwrap();
        let bf16_type = context.bfloat16_type().as_ref().cast::<FloatTypeRef>().unwrap();
        let tf32_type = context.floattf32_type().as_ref().cast::<FloatTypeRef>().unwrap();

        assert_eq!(context.stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::Default), None);

        let attribute = context
            .stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::any_f8_any_f8_f32(f8e4m3fn_type, f8e5m2_type))
            .unwrap();
        assert_eq!(attribute.lhs_precision_type(), f8e4m3fn_type);
        assert_eq!(attribute.rhs_precision_type(), f8e5m2_type);
        assert_eq!(attribute.accumulation_type(), f32_type);
        assert_eq!(attribute.primitive_operation_count(), 1);
        assert_eq!(attribute.allow_imprecise_accumulation(), false);

        let attribute = context
            .stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::any_f8_any_f8_f32_fast_accumulation(
                f8e4m3fn_type,
                f8e5m2_type,
            ))
            .unwrap();
        assert_eq!(attribute.lhs_precision_type(), f8e4m3fn_type);
        assert_eq!(attribute.rhs_precision_type(), f8e5m2_type);
        assert_eq!(attribute.accumulation_type(), f32_type);
        assert_eq!(attribute.primitive_operation_count(), 1);
        assert_eq!(attribute.allow_imprecise_accumulation(), true);

        let attribute = context
            .stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::any_f8_any_f8_any(
                f8e4m3fn_type,
                f8e5m2_type,
                f32_type,
            ))
            .unwrap();
        assert_eq!(attribute.lhs_precision_type(), f8e4m3fn_type);
        assert_eq!(attribute.rhs_precision_type(), f8e5m2_type);
        assert_eq!(attribute.accumulation_type(), f32_type);
        assert_eq!(attribute.primitive_operation_count(), 1);
        assert_eq!(attribute.allow_imprecise_accumulation(), false);

        let attribute = context
            .stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::any_f8_any_f8_any_fast_accumulation(
                f8e4m3fn_type,
                f8e5m2_type,
                f32_type,
            ))
            .unwrap();
        assert_eq!(attribute.lhs_precision_type(), f8e4m3fn_type);
        assert_eq!(attribute.rhs_precision_type(), f8e5m2_type);
        assert_eq!(attribute.accumulation_type(), f32_type);
        assert_eq!(attribute.primitive_operation_count(), 1);
        assert_eq!(attribute.allow_imprecise_accumulation(), true);

        let attribute = context.stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::f16_f16_f16()).unwrap();
        assert_eq!(attribute.lhs_precision_type(), f16_type);
        assert_eq!(attribute.rhs_precision_type(), f16_type);
        assert_eq!(attribute.accumulation_type(), f16_type);
        assert_eq!(attribute.lhs_component_count(), 1);
        assert_eq!(attribute.rhs_component_count(), 1);
        assert_eq!(attribute.primitive_operation_count(), 1);
        assert_eq!(attribute.allow_imprecise_accumulation(), false);

        let attribute = context.stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::f16_f16_f32()).unwrap();
        assert_eq!(attribute.lhs_precision_type(), f16_type);
        assert_eq!(attribute.rhs_precision_type(), f16_type);
        assert_eq!(attribute.accumulation_type(), f32_type);
        assert_eq!(attribute.lhs_component_count(), 1);
        assert_eq!(attribute.rhs_component_count(), 1);
        assert_eq!(attribute.primitive_operation_count(), 1);
        assert_eq!(attribute.allow_imprecise_accumulation(), false);

        let attribute = context.stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::bf16_bf16_bf16()).unwrap();
        assert_eq!(attribute.lhs_precision_type(), bf16_type);
        assert_eq!(attribute.rhs_precision_type(), bf16_type);
        assert_eq!(attribute.accumulation_type(), bf16_type);
        assert_eq!(attribute.primitive_operation_count(), 1);
        assert_eq!(attribute.allow_imprecise_accumulation(), false);

        let attribute = context.stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::bf16_bf16_f32()).unwrap();
        assert_eq!(attribute.lhs_precision_type(), bf16_type);
        assert_eq!(attribute.rhs_precision_type(), bf16_type);
        assert_eq!(attribute.accumulation_type(), f32_type);
        assert_eq!(attribute.primitive_operation_count(), 1);
        assert_eq!(attribute.allow_imprecise_accumulation(), false);

        let attribute = context.stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::bf16_bf16_f32_x3()).unwrap();
        assert_eq!(attribute.lhs_precision_type(), bf16_type);
        assert_eq!(attribute.rhs_precision_type(), bf16_type);
        assert_eq!(attribute.accumulation_type(), f32_type);
        assert_eq!(attribute.primitive_operation_count(), 3);
        assert_eq!(attribute.allow_imprecise_accumulation(), false);

        let attribute = context.stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::bf16_bf16_f32_x6()).unwrap();
        assert_eq!(attribute.lhs_precision_type(), bf16_type);
        assert_eq!(attribute.rhs_precision_type(), bf16_type);
        assert_eq!(attribute.accumulation_type(), f32_type);
        assert_eq!(attribute.primitive_operation_count(), 6);
        assert_eq!(attribute.allow_imprecise_accumulation(), false);

        let attribute = context.stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::bf16_bf16_f32_x9()).unwrap();
        assert_eq!(attribute.lhs_precision_type(), bf16_type);
        assert_eq!(attribute.rhs_precision_type(), bf16_type);
        assert_eq!(attribute.accumulation_type(), f32_type);
        assert_eq!(attribute.primitive_operation_count(), 9);
        assert_eq!(attribute.allow_imprecise_accumulation(), false);

        let attribute = context.stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::tf32_tf32_f32()).unwrap();
        assert_eq!(attribute.lhs_precision_type(), tf32_type);
        assert_eq!(attribute.rhs_precision_type(), tf32_type);
        assert_eq!(attribute.accumulation_type(), f32_type);
        assert_eq!(attribute.primitive_operation_count(), 1);
        assert_eq!(attribute.allow_imprecise_accumulation(), false);

        let attribute = context.stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::tf32_tf32_f32_x3()).unwrap();
        assert_eq!(attribute.lhs_precision_type(), tf32_type);
        assert_eq!(attribute.rhs_precision_type(), tf32_type);
        assert_eq!(attribute.accumulation_type(), f32_type);
        assert_eq!(attribute.primitive_operation_count(), 3);
        assert_eq!(attribute.allow_imprecise_accumulation(), false);

        let attribute = context.stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::f32_f32_f32()).unwrap();
        assert_eq!(attribute.lhs_precision_type(), f32_type);
        assert_eq!(attribute.rhs_precision_type(), f32_type);
        assert_eq!(attribute.accumulation_type(), f32_type);
        assert_eq!(attribute.primitive_operation_count(), 1);
        assert_eq!(attribute.allow_imprecise_accumulation(), false);

        let attribute = context.stable_hlo_dot_algorithm_from_preset(DotAlgorithmPreset::f64_f64_f64()).unwrap();
        assert_eq!(attribute.lhs_precision_type(), f64_type);
        assert_eq!(attribute.rhs_precision_type(), f64_type);
        assert_eq!(attribute.accumulation_type(), f64_type);
        assert_eq!(attribute.primitive_operation_count(), 1);
        assert_eq!(attribute.allow_imprecise_accumulation(), false);
    }

    #[test]
    fn test_dot_general() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let lhs_type = context
            .tensor_type(i64_type, &[Size::Static(2), Size::Static(2), Size::Static(2)], None, location)
            .unwrap();
        let rhs_type = context
            .tensor_type(i64_type, &[Size::Static(2), Size::Static(2), Size::Static(2)], None, location)
            .unwrap();
        let result_type = context
            .tensor_type(i64_type, &[Size::Static(2), Size::Static(2), Size::Static(2)], None, location)
            .unwrap();
        let dimensions = context.stable_hlo_dot_dimensions(&[0], &[0], &[2], &[1]);
        let algorithm = context.stable_hlo_dot_algorithm(
            context.float8e5m2_type(),
            context.float8e5m2_type(),
            context.float32_type(),
            1,
            1,
            1,
            true,
        );
        module.body().append_operation({
            let mut block = context.block(&[(lhs_type, location), (rhs_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = dot_general(lhs, rhs, dimensions, None, Some(algorithm), result_type, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.dimensions(), dimensions);
            assert_eq!(op.precision(), None);
            assert_eq!(op.algorithm(), Some(algorithm));
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);

            // Test the precision attribute using a dummy op.
            let dummy_op = dot_general(
                lhs,
                rhs,
                dimensions,
                Some((Precision::High, Precision::Highest)),
                None,
                result_type,
                location,
            );
            assert_eq!(dummy_op.lhs(), lhs);
            assert_eq!(dummy_op.rhs(), rhs);
            assert_eq!(dummy_op.dimensions(), dimensions);
            assert_eq!(dummy_op.precision(), Some((Precision::High, Precision::Highest)));
            assert_eq!(dummy_op.algorithm(), None);
            assert_eq!(dummy_op.operands().count(), 2);
            assert_eq!(dummy_op.results().count(), 1);

            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "dot_general_test",
                func::FuncAttributes {
                    arguments: vec![lhs_type.into(), rhs_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @dot_general_test(%arg0: tensor<2x2x2xi64>, %arg1: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {
                    %0 = stablehlo.dot_general \
                      %arg0, \
                      %arg1, \
                      batching_dims = [0] x [0], \
                      contracting_dims = [2] x [1], \
                      algorithm = <\
                        lhs_precision_type = f8E5M2, \
                        rhs_precision_type = f8E5M2, \
                        accumulation_type = f32, \
                        lhs_component_count = 1, \
                        rhs_component_count = 1, \
                        num_primitive_operations = 1, \
                        allow_imprecise_accumulation = true\
                      > : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
                    return %0 : tensor<2x2x2xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_convolution_dimensions_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_convolution_dimensions(0, 3, &[1, 2], 2, 3, &[0, 1], 0, 3, &[1, 2]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.input_batch_dimension(), 0);
        assert_eq!(attribute.input_feature_dimension(), 3);
        assert_eq!(attribute.input_spatial_dimensions(), vec![1, 2]);
        assert_eq!(attribute.kernel_input_feature_dimension(), 2);
        assert_eq!(attribute.kernel_output_feature_dimension(), 3);
        assert_eq!(attribute.kernel_spatial_dimensions(), vec![0, 1]);
        assert_eq!(attribute.output_batch_dimension(), 0);
        assert_eq!(attribute.output_feature_dimension(), 3);
        assert_eq!(attribute.output_spatial_dimensions(), vec![1, 2]);
    }

    #[test]
    fn test_convolution_dimensions_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_convolution_dimensions(0, 3, &[1, 2], 2, 3, &[0, 1], 0, 3, &[1, 2]);
        let attribute_2 = context.stable_hlo_convolution_dimensions(0, 3, &[1, 2], 2, 3, &[0, 1], 0, 3, &[1, 2]);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_convolution_dimensions(1, 0, &[2, 3], 4, 5, &[6, 7], 8, 9, &[10, 11]);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_convolution_dimensions(0, 3, &[1, 2], 2, 3, &[0, 1], 0, 3, &[1, 2]);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_convolution_dimensions_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_convolution_dimensions(0, 3, &[1, 2], 2, 3, &[0, 1], 0, 3, &[1, 2]);
        test_attribute_display_and_debug(attribute, "#stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>");
    }

    #[test]
    fn test_convolution_dimensions_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_convolution_dimensions(0, 3, &[1, 2], 2, 3, &[0, 1], 0, 3, &[1, 2]);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_convolution() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let input_type = context
            .tensor_type(
                f32_type,
                &[Size::Static(32), Size::Static(4), Size::Static(4), Size::Static(16)],
                None,
                location,
            )
            .unwrap();
        let kernel_type = context
            .tensor_type(
                f32_type,
                &[Size::Static(3), Size::Static(3), Size::Static(16), Size::Static(4)],
                None,
                location,
            )
            .unwrap();
        let result_type = context
            .tensor_type(
                f32_type,
                &[Size::Static(16), Size::Static(8), Size::Static(4), Size::Static(4)],
                None,
                location,
            )
            .unwrap();
        let dimensions = context.stable_hlo_convolution_dimensions(0, 3, &[1, 2], 2, 3, &[0, 1], 0, 3, &[1, 2]);
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (kernel_type, location)]);
            let input = block.argument(0).unwrap();
            let kernel = block.argument(1).unwrap();
            let op = convolution(
                input,
                kernel,
                dimensions,
                2,
                1,
                Some(&[1, 1]),
                Some(&[(4, 2), (0, 2)]),
                Some(&[1, 1]),
                Some(&[1, 1]),
                Some(&[false, false]),
                Some((Precision::Default, Precision::Highest)),
                result_type,
                location,
            );
            assert_eq!(op.input(), input);
            assert_eq!(op.kernel(), kernel);
            assert_eq!(op.dimensions(), dimensions);
            assert_eq!(op.batch_group_count(), 2);
            assert_eq!(op.feature_group_count(), 1);
            assert_eq!(op.window_strides(), Some(vec![1, 1]));
            assert_eq!(op.padding(), Some(vec![(4, 2), (0, 2)]));
            assert_eq!(op.lhs_dilation(), Some(vec![1, 1]));
            assert_eq!(op.rhs_dilation(), Some(vec![1, 1]));
            assert_eq!(op.window_reversal(), Some(vec![false, false]));
            assert_eq!(op.precision(), Some((Precision::Default, Precision::Highest)));
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "convolution_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), kernel_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @convolution_test(\
                    %arg0: tensor<32x4x4x16xf32>, \
                    %arg1: tensor<3x3x16x4xf32>\
                  ) -> tensor<16x8x4x4xf32> {
                    %0 = stablehlo.convolution(%arg0, %arg1) \
                      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], \
                      window = {\
                        stride = [1, 1], \
                        pad = [[4, 2], [0, 2]], \
                        lhs_dilate = [1, 1], \
                        rhs_dilate = [1, 1], \
                        reverse = [false, false]\
                      } {\
                        batch_group_count = 2 : i64, \
                        feature_group_count = 1 : i64, \
                        precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision HIGHEST>]\
                      } : (tensor<32x4x4x16xf32>, tensor<3x3x16x4xf32>) -> tensor<16x8x4x4xf32>
                    return %0 : tensor<16x8x4x4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_dynamic_convolution() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let i64_type = context.signless_integer_type(64);
        let input_type = context
            .tensor_type(
                f32_type,
                &[Size::Static(32), Size::Static(4), Size::Static(4), Size::Static(16)],
                None,
                location,
            )
            .unwrap();
        let kernel_type = context
            .tensor_type(
                f32_type,
                &[Size::Static(3), Size::Static(3), Size::Static(16), Size::Static(4)],
                None,
                location,
            )
            .unwrap();
        let padding_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let result_type = context
            .tensor_type(
                f32_type,
                &[Size::Static(16), Size::Static(8), Size::Static(4), Size::Static(4)],
                None,
                location,
            )
            .unwrap();
        let dimensions = context.stable_hlo_convolution_dimensions(0, 3, &[1, 2], 2, 3, &[0, 1], 0, 3, &[1, 2]);
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (kernel_type, location), (padding_type, location)]);
            let input = block.argument(0).unwrap();
            let kernel = block.argument(1).unwrap();
            let padding = block.argument(2).unwrap();
            let op = dynamic_convolution(
                input,
                kernel,
                padding,
                dimensions,
                2,
                1,
                Some(&[1, 1]),
                Some(&[1, 1]),
                Some(&[1, 1]),
                Some(&[false, false]),
                Some((Precision::Default, Precision::Highest)),
                result_type,
                location,
            );
            assert_eq!(op.padding(), padding);
            assert_eq!(op.dimensions(), dimensions);
            assert_eq!(op.batch_group_count(), 2);
            assert_eq!(op.feature_group_count(), 1);
            assert_eq!(op.window_strides(), Some(vec![1, 1]));
            assert_eq!(op.lhs_dilation(), Some(vec![1, 1]));
            assert_eq!(op.rhs_dilation(), Some(vec![1, 1]));
            assert_eq!(op.window_reversal(), Some(vec![false, false]));
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "dynamic_conv_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), kernel_type.into(), padding_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @dynamic_conv_test(\
                    %arg0: tensor<32x4x4x16xf32>, \
                    %arg1: tensor<3x3x16x4xf32>, \
                    %arg2: tensor<2x2xi64>\
                  ) -> tensor<16x8x4x4xf32> {
                    %0 = \"stablehlo.dynamic_conv\"(%arg0, %arg1, %arg2) <{\
                      batch_group_count = 2 : i64, \
                      dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, \
                      feature_group_count = 1 : i64, \
                      lhs_dilation = array<i64: 1, 1>, \
                      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision HIGHEST>], \
                      rhs_dilation = array<i64: 1, 1>, \
                      window_reversal = array<i1: false, false>, \
                      window_strides = array<i64: 1, 1>\
                    }> : (tensor<32x4x4x16xf32>, tensor<3x3x16x4xf32>, tensor<2x2xi64>) -> tensor<16x8x4x4xf32>
                    return %0 : tensor<16x8x4x4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_cholesky() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let matrix_type = context.tensor_type(f32_type, &[Size::Static(3), Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(matrix_type, location)]);
            let input = block.argument(0).unwrap();
            let op = cholesky(input, true, location);
            assert_eq!(op.lower(), true);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "cholesky_test",
                func::FuncAttributes {
                    arguments: vec![matrix_type.into()],
                    results: vec![matrix_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @cholesky_test(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
                    %0 = stablehlo.cholesky %arg0, lower = true : tensor<3x3xf32>
                    return %0 : tensor<3x3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_triangular_solve_transpose_type_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_triangular_solve_transpose_type(TriangularSolveTransposeType::Adjoint);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), TriangularSolveTransposeType::Adjoint);
    }

    #[test]
    fn test_triangular_solve_transpose_type_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_triangular_solve_transpose_type(TriangularSolveTransposeType::Adjoint);
        let attribute_2 = context.stable_hlo_triangular_solve_transpose_type(TriangularSolveTransposeType::Adjoint);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_triangular_solve_transpose_type(TriangularSolveTransposeType::NoTranspose);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_triangular_solve_transpose_type(TriangularSolveTransposeType::Adjoint);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_triangular_solve_transpose_type_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_triangular_solve_transpose_type(TriangularSolveTransposeType::Adjoint);
        test_attribute_display_and_debug(attribute, "#stablehlo<transpose ADJOINT>");
    }

    #[test]
    fn test_triangular_solve_transpose_type_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_triangular_solve_transpose_type(TriangularSolveTransposeType::Adjoint);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_triangular_solve() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let matrix_type = context.tensor_type(f32_type, &[Size::Static(3), Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(matrix_type, location), (matrix_type, location)]);
            let a = block.argument(0).unwrap();
            let b = block.argument(1).unwrap();
            let op = triangular_solve(a, b, true, true, false, TriangularSolveTransposeType::NoTranspose, location);
            assert_eq!(op.a(), a);
            assert_eq!(op.b(), b);
            assert_eq!(op.left_side(), true);
            assert_eq!(op.lower(), true);
            assert_eq!(op.unit_diagonal(), false);
            assert_eq!(op.transpose_a(), TriangularSolveTransposeType::NoTranspose);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "triangular_solve_test",
                func::FuncAttributes {
                    arguments: vec![matrix_type.into(), matrix_type.into()],
                    results: vec![matrix_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @triangular_solve_test(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
                    %0 = \"stablehlo.triangular_solve\"(%arg0, %arg1) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
                    return %0 : tensor<3x3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_fft_type_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_fft_type(FftType::RFFT);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), FftType::RFFT);
    }

    #[test]
    fn test_fft_type_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_fft_type(FftType::RFFT);
        let attribute_2 = context.stable_hlo_fft_type(FftType::RFFT);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_fft_type(FftType::FFT);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_fft_type(FftType::RFFT);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_fft_type_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_fft_type(FftType::RFFT);
        test_attribute_display_and_debug(attribute, "#stablehlo<fft_type RFFT>");
    }

    #[test]
    fn test_fft_type_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_fft_type(FftType::RFFT);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_fft() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let tensor_type = context
            .tensor_type(context.complex_type(context.float32_type()), &[Size::Static(4)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let fft_op = fft(input, FftType::FFT, &[4], location);
            assert_eq!(fft_op.fft_type(), FftType::FFT);
            assert_eq!(fft_op.fft_length(), vec![4]);
            assert_eq!(fft_op.operands().count(), 1);
            assert_eq!(fft_op.results().count(), 1);
            let fft_block = block.append_operation(fft_op);
            block.append_operation(func::r#return(&[fft_block.result(0).unwrap()], location));
            func::func(
                "fft_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into()],
                    results: vec![tensor_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @fft_test(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
                    %0 = stablehlo.fft %arg0, \
                      type =  FFT, \
                      length = [4] \
                    : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
                    return %0 : tensor<4xcomplex<f32>>
                  }
                }
            "},
        );
    }
}
