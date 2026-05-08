use ryft_xla_sys::bindings::MlirAttribute;
use ryft_xla_sys::mlir::dialects::mosaic::gpu::{
    MlirMosaicGpuEnumAttribute, mlirAttributeIsAMosaicGpuEnumAttr, mlirAttributeIsAMosaicGpuTmemAttr,
    mlirMosaicGpuCopyPartitionedAttrGet, mlirMosaicGpuCopyPartitionedAttrGetAxis, mlirMosaicGpuCopyReplicatedAttrGet,
    mlirMosaicGpuEnumAttrGet, mlirMosaicGpuEnumAttrGetValue, mlirMosaicGpuIsACopyPartitionAttr,
    mlirMosaicGpuIsACopyPartitionedAttr, mlirMosaicGpuIsACopyReplicatedAttr, mlirMosaicGpuIsAReplicatedAttr,
    mlirMosaicGpuIsASwizzleTransformAttr, mlirMosaicGpuIsATileTransformAttr, mlirMosaicGpuIsATiledLayoutAttr,
    mlirMosaicGpuIsATransposeTransformAttr, mlirMosaicGpuIsAWGSplatFragLayoutAttr,
    mlirMosaicGpuIsAWGStridedFragLayoutAttr, mlirMosaicGpuReplicatedAttrGet, mlirMosaicGpuReplicatedAttrGetTimes,
    mlirMosaicGpuSwizzleTransformAttrGet, mlirMosaicGpuSwizzleTransformAttrGetSwizzle,
    mlirMosaicGpuTileTransformAttrGet, mlirMosaicGpuTileTransformAttrGetTiling, mlirMosaicGpuTiledLayoutAttrGet,
    mlirMosaicGpuTiledLayoutAttrGetLaneDims, mlirMosaicGpuTiledLayoutAttrGetTiling,
    mlirMosaicGpuTiledLayoutAttrGetVectorDim, mlirMosaicGpuTiledLayoutAttrGetWarpDims, mlirMosaicGpuTmemAttrGet,
    mlirMosaicGpuTransposeTransformAttrGet, mlirMosaicGpuTransposeTransformAttrGetPermutation,
    mlirMosaicGpuWGSplatFragLayoutAttrGet, mlirMosaicGpuWGSplatFragLayoutAttrGetShape,
    mlirMosaicGpuWGStridedFragLayoutAttrGet, mlirMosaicGpuWGStridedFragLayoutAttrGetShape,
    mlirMosaicGpuWGStridedFragLayoutAttrGetVectorSize,
};

use crate::{
    ArrayAttributeRef, Attribute, Context, DenseInteger32ArrayAttributeRef, DenseInteger64ArrayAttributeRef,
    DialectHandle, Error, StringRef, mlir_subtype_trait_impls,
};

macro_rules! mosaic_gpu_enum_attribute {
    (
        enum_name = $enum_name:ident,
        attribute_name = $attribute_name:ident,
        context_method = $context_method:ident,
        ffi_kind = $ffi_kind:path,
        mnemonic = $mnemonic:literal,
        description = $description:literal,
        variants = { $($variant:ident => ($value:literal, $integer:literal)),+ $(,)* } $(,)*
    ) => {
        #[doc = "Mosaic GPU "]
        #[doc = $description]
        #[doc = "."]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub enum $enum_name {
            $($variant,)+
        }

        impl $enum_name {
            /// Returns the MLIR spelling of this enum value.
            pub fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $value,)+
                }
            }

            /// Returns the integer value associated with this enum value in the Mosaic GPU dialect.
            pub fn as_i32(&self) -> i32 {
                match self {
                    $(Self::$variant => $integer,)+
                }
            }

            /// Parses the MLIR spelling of this enum value.
            pub fn from_str(value: &str) -> Option<Self> {
                match value {
                    $($value => Some(Self::$variant),)+
                    _ => None,
                }
            }

            /// Parses the integer value associated with this enum value.
            pub fn from_i32(value: i32) -> Option<Self> {
                match value {
                    $($integer => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }

        #[doc = "Mosaic GPU "]
        #[doc = $description]
        #[doc = " [`Attribute`]."]
        #[derive(Copy, Clone)]
        pub struct $attribute_name<'c, 't> {
            /// Handle that represents this [`Attribute`] in the MLIR C API.
            handle: MlirAttribute,

            /// [`Context`] that owns this [`Attribute`].
            context: &'c Context<'t>,
        }

        impl<'c, 't> $attribute_name<'c, 't> {
            /// Returns the enum value stored in this attribute.
            pub fn value(&self) -> Result<$enum_name, Error> {
                let value = unsafe { StringRef::from_c_api(mlirMosaicGpuEnumAttrGetValue(self.handle, $ffi_kind)) };
                value
                    .as_str()
                    .ok()
                    .and_then($enum_name::from_str)
                    .ok_or_else(|| Error::invalid_argument(concat!("invalid Mosaic GPU `", $mnemonic, "` attribute")))
            }
        }

        impl<'c, 't> Attribute<'c, 't> for $attribute_name<'c, 't> {
            unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
                if handle.ptr.is_null() {
                    return Err(Error::internal("expected non-null MLIR attribute handle"));
                }
                if unsafe { mlirAttributeIsAMosaicGpuEnumAttr(handle, $ffi_kind) } {
                    Ok(Self { handle, context })
                } else {
                    Err(Error::invalid_argument("expected MLIR attribute handle"))
                }
            }

            unsafe fn to_c_api(&self) -> MlirAttribute {
                self.handle
            }

            fn context(&self) -> &'c Context<'t> {
                self.context
            }
        }

        mlir_subtype_trait_impls!($attribute_name<'c, 't> as Attribute, mlir_type = Attribute);

        impl<'t> Context<'t> {
            #[doc = "Creates a Mosaic GPU "]
            #[doc = $description]
            #[doc = " attribute owned by this [`Context`]."]
            pub fn $context_method<'c>(&'c self, value: $enum_name) -> Result<$attribute_name<'c, 't>, Error> {
                self.load_dialect(DialectHandle::mosaic_gpu()?)?;
                let value = StringRef::from(value.as_str());
                Ok(unsafe {
                    $attribute_name {
                        handle: mlirMosaicGpuEnumAttrGet(*self.handle.borrow_mut(), $ffi_kind, value.to_c_api()),
                        context: self,
                    }
                })
            }
        }
    };
}

/// Mosaic GPU warpgroup strided fragment layout [`Attribute`].
#[derive(Copy, Clone)]
pub struct WgStridedFragLayoutAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> WgStridedFragLayoutAttributeRef<'c, 't> {
    /// Returns the logical array shape described by this layout.
    pub fn shape(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        unsafe {
            DenseInteger64ArrayAttributeRef::from_c_api(
                mlirMosaicGpuWGStridedFragLayoutAttrGetShape(self.handle),
                self.context,
            )
            .map_err(|_| Error::internal("expected non-null Mosaic GPU warpgroup strided fragment shape"))
        }
    }

    /// Returns the number of contiguous elements assigned to each thread.
    pub fn vector_size(&self) -> i32 {
        unsafe { mlirMosaicGpuWGStridedFragLayoutAttrGetVectorSize(self.handle) }
    }
}

impl<'c, 't> Attribute<'c, 't> for WgStridedFragLayoutAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirMosaicGpuIsAWGStridedFragLayoutAttr(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(WgStridedFragLayoutAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Mosaic GPU warpgroup splat fragment layout [`Attribute`].
#[derive(Copy, Clone)]
pub struct WgSplatFragLayoutAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> WgSplatFragLayoutAttributeRef<'c, 't> {
    /// Returns the shape that the scalar value is splatted to.
    pub fn shape(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        unsafe {
            DenseInteger64ArrayAttributeRef::from_c_api(
                mlirMosaicGpuWGSplatFragLayoutAttrGetShape(self.handle),
                self.context,
            )
            .map_err(|_| Error::internal("expected non-null Mosaic GPU warpgroup splat fragment shape"))
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for WgSplatFragLayoutAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirMosaicGpuIsAWGSplatFragLayoutAttr(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(WgSplatFragLayoutAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Mosaic GPU replicated dimension [`Attribute`] used in tiled layouts.
#[derive(Copy, Clone)]
pub struct ReplicatedAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl ReplicatedAttributeRef<'_, '_> {
    /// Returns the replication count.
    pub fn times(&self) -> i32 {
        unsafe { mlirMosaicGpuReplicatedAttrGetTimes(self.handle) }
    }
}

impl<'c, 't> Attribute<'c, 't> for ReplicatedAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirMosaicGpuIsAReplicatedAttr(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(ReplicatedAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Mosaic GPU tiled layout [`Attribute`].
#[derive(Copy, Clone)]
pub struct TiledLayoutAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> TiledLayoutAttributeRef<'c, 't> {
    /// Returns the tiling expression.
    pub fn tiling(&self) -> Result<ArrayAttributeRef<'c, 't>, Error> {
        unsafe {
            ArrayAttributeRef::from_c_api(mlirMosaicGpuTiledLayoutAttrGetTiling(self.handle), self.context)
                .map_err(|_| Error::internal("expected non-null Mosaic GPU tiled layout tiling attribute"))
        }
    }

    /// Returns the warp dimensions.
    pub fn warp_dims(&self) -> Result<ArrayAttributeRef<'c, 't>, Error> {
        unsafe {
            ArrayAttributeRef::from_c_api(mlirMosaicGpuTiledLayoutAttrGetWarpDims(self.handle), self.context)
                .map_err(|_| Error::internal("expected non-null Mosaic GPU tiled layout warp dimensions"))
        }
    }

    /// Returns the lane dimensions.
    pub fn lane_dims(&self) -> Result<ArrayAttributeRef<'c, 't>, Error> {
        unsafe {
            ArrayAttributeRef::from_c_api(mlirMosaicGpuTiledLayoutAttrGetLaneDims(self.handle), self.context)
                .map_err(|_| Error::internal("expected non-null Mosaic GPU tiled layout lane dimensions"))
        }
    }

    /// Returns the vector dimension.
    pub fn vector_dim(&self) -> i32 {
        unsafe { mlirMosaicGpuTiledLayoutAttrGetVectorDim(self.handle) }
    }
}

impl<'c, 't> Attribute<'c, 't> for TiledLayoutAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirMosaicGpuIsATiledLayoutAttr(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(TiledLayoutAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

mosaic_gpu_enum_attribute!(
    enum_name = Dimension,
    attribute_name = DimensionAttributeRef,
    context_method = mosaic_gpu_dimension_attribute,
    ffi_kind = MlirMosaicGpuEnumAttribute::RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_DIMENSION,
    mnemonic = "dimension",
    description = "dimension",
    variants = {
        X => ("x", 0),
        Y => ("y", 1),
        Z => ("z", 2),
    },
);

mosaic_gpu_enum_attribute!(
    enum_name = SwizzlingMode,
    attribute_name = SwizzlingModeAttributeRef,
    context_method = mosaic_gpu_swizzling_mode_attribute,
    ffi_kind = MlirMosaicGpuEnumAttribute::RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_SWIZZLING_MODE,
    mnemonic = "swizzling_mode",
    description = "swizzling mode",
    variants = {
        NoSwizzle => ("kNoSwizzle", 16),
        Swizzle32Byte => ("k32ByteSwizzle", 32),
        Swizzle64Byte => ("k64ByteSwizzle", 64),
        Swizzle128Byte => ("k128ByteSwizzle", 128),
    },
);

mosaic_gpu_enum_attribute!(
    enum_name = TmaReduction,
    attribute_name = TmaReductionAttributeRef,
    context_method = mosaic_gpu_tma_reduction_attribute,
    ffi_kind = MlirMosaicGpuEnumAttribute::RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_TMA_REDUCTION,
    mnemonic = "tma_reduction",
    description = "TMA reduction operation",
    variants = {
        Add => ("add", 0),
        Min => ("min", 1),
        Max => ("max", 2),
        Increment => ("inc", 3),
        Decrement => ("dec", 4),
        And => ("and", 5),
        Or => ("or", 6),
        Xor => ("xor", 7),
        UnsignedMin => ("umin", 8),
        UnsignedMax => ("umax", 9),
        SignedMin => ("smin", 10),
        SignedMax => ("smax", 11),
    },
);

mosaic_gpu_enum_attribute!(
    enum_name = OobFillMode,
    attribute_name = OobFillModeAttributeRef,
    context_method = mosaic_gpu_oob_fill_mode_attribute,
    ffi_kind = MlirMosaicGpuEnumAttribute::RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_OOB_FILL_MODE,
    mnemonic = "oob_fill_mode",
    description = "out-of-bounds fill mode",
    variants = {
        Undefined => ("undefined", 0),
        PromiseInBounds => ("promise_in_bounds", 1),
        Zeros => ("zeros", 2),
    },
);

mosaic_gpu_enum_attribute!(
    enum_name = MultimemLoadReductionType,
    attribute_name = MultimemLoadReductionTypeAttributeRef,
    context_method = mosaic_gpu_multimem_load_reduction_type_attribute,
    ffi_kind = MlirMosaicGpuEnumAttribute::RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_MULTIMEM_LOAD_REDUCTION_TYPE,
    mnemonic = "multimem_load_reduction_type",
    description = "multimem load reduction type",
    variants = {
        Add => ("add", 0),
        Min => ("min", 1),
        Max => ("max", 2),
        And => ("and", 3),
        Or => ("or", 4),
        Xor => ("xor", 5),
        UnsignedMin => ("umin", 6),
        UnsignedMax => ("umax", 7),
        SignedMin => ("smin", 8),
        SignedMax => ("smax", 9),
    },
);

mosaic_gpu_enum_attribute!(
    enum_name = AtomicOpType,
    attribute_name = AtomicOpTypeAttributeRef,
    context_method = mosaic_gpu_atomic_op_type_attribute,
    ffi_kind = MlirMosaicGpuEnumAttribute::RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_ATOMIC_OP_TYPE,
    mnemonic = "atomic_op_type",
    description = "atomic operation type",
    variants = {
        Add => ("add", 0),
        Min => ("min", 1),
        Max => ("max", 2),
        And => ("and", 3),
        Or => ("or", 4),
        Xor => ("xor", 5),
    },
);

/// Mosaic GPU tile transform [`Attribute`] for shared-memory memrefs.
#[derive(Copy, Clone)]
pub struct TileTransformAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> TileTransformAttributeRef<'c, 't> {
    /// Returns the tiling factors.
    pub fn tiling(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        unsafe {
            DenseInteger32ArrayAttributeRef::from_c_api(
                mlirMosaicGpuTileTransformAttrGetTiling(self.handle),
                self.context,
            )
            .map_err(|_| Error::internal("expected non-null Mosaic GPU tile transform tiling attribute"))
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for TileTransformAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirMosaicGpuIsATileTransformAttr(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(TileTransformAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Mosaic GPU transpose transform [`Attribute`] for shared-memory memrefs.
#[derive(Copy, Clone)]
pub struct TransposeTransformAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> TransposeTransformAttributeRef<'c, 't> {
    /// Returns the permutation.
    pub fn permutation(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        unsafe {
            DenseInteger32ArrayAttributeRef::from_c_api(
                mlirMosaicGpuTransposeTransformAttrGetPermutation(self.handle),
                self.context,
            )
            .map_err(|_| Error::internal("expected non-null Mosaic GPU transpose transform permutation"))
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for TransposeTransformAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirMosaicGpuIsATransposeTransformAttr(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(TransposeTransformAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Mosaic GPU swizzle transform [`Attribute`] for shared-memory memrefs.
#[derive(Copy, Clone)]
pub struct SwizzleTransformAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl SwizzleTransformAttributeRef<'_, '_> {
    /// Returns the swizzling mode.
    pub fn swizzle(&self) -> Result<SwizzlingMode, Error> {
        let value = unsafe { mlirMosaicGpuSwizzleTransformAttrGetSwizzle(self.handle) };
        SwizzlingMode::from_i32(value)
            .ok_or_else(|| Error::invalid_argument("invalid Mosaic GPU swizzle transform attribute"))
    }
}

impl<'c, 't> Attribute<'c, 't> for SwizzleTransformAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirMosaicGpuIsASwizzleTransformAttr(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(SwizzleTransformAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Mosaic GPU copy partition strategy [`Attribute`].
#[derive(Copy, Clone)]
pub struct CopyPartitionAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for CopyPartitionAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirMosaicGpuIsACopyPartitionAttr(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(CopyPartitionAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Mosaic GPU replicated copy partition [`Attribute`].
#[derive(Copy, Clone)]
pub struct CopyReplicatedAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for CopyReplicatedAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirMosaicGpuIsACopyReplicatedAttr(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(CopyReplicatedAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Mosaic GPU partitioned copy partition [`Attribute`].
#[derive(Copy, Clone)]
pub struct CopyPartitionedAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl CopyPartitionedAttributeRef<'_, '_> {
    /// Returns the cluster axis along which this copy is partitioned.
    pub fn axis(&self) -> i32 {
        unsafe { mlirMosaicGpuCopyPartitionedAttrGetAxis(self.handle) }
    }
}

impl<'c, 't> Attribute<'c, 't> for CopyPartitionedAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirMosaicGpuIsACopyPartitionedAttr(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(CopyPartitionedAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Mosaic GPU tensor-memory address-space [`Attribute`].
#[derive(Copy, Clone)]
pub struct TmemAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for TmemAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsAMosaicGpuTmemAttr(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(TmemAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a Mosaic GPU warpgroup strided fragment layout attribute owned by this [`Context`].
    pub fn mosaic_gpu_wg_strided_frag_layout_attribute<'c>(
        &'c self,
        shape: DenseInteger64ArrayAttributeRef<'c, 't>,
        vector_size: i32,
    ) -> Result<WgStridedFragLayoutAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_gpu()?)?;
        Ok(unsafe {
            WgStridedFragLayoutAttributeRef {
                handle: mlirMosaicGpuWGStridedFragLayoutAttrGet(
                    *self.handle.borrow_mut(),
                    shape.to_c_api(),
                    vector_size,
                ),
                context: self,
            }
        })
    }

    /// Creates a Mosaic GPU warpgroup splat fragment layout attribute owned by this [`Context`].
    pub fn mosaic_gpu_wg_splat_frag_layout_attribute<'c>(
        &'c self,
        shape: DenseInteger64ArrayAttributeRef<'c, 't>,
    ) -> Result<WgSplatFragLayoutAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_gpu()?)?;
        Ok(unsafe {
            WgSplatFragLayoutAttributeRef {
                handle: mlirMosaicGpuWGSplatFragLayoutAttrGet(*self.handle.borrow_mut(), shape.to_c_api()),
                context: self,
            }
        })
    }

    /// Creates a Mosaic GPU replicated dimension attribute owned by this [`Context`].
    pub fn mosaic_gpu_replicated_attribute<'c>(&'c self, times: i32) -> Result<ReplicatedAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_gpu()?)?;
        Ok(unsafe {
            ReplicatedAttributeRef {
                handle: mlirMosaicGpuReplicatedAttrGet(*self.handle.borrow_mut(), times),
                context: self,
            }
        })
    }

    /// Creates a Mosaic GPU tiled layout attribute owned by this [`Context`].
    pub fn mosaic_gpu_tiled_layout_attribute<'c>(
        &'c self,
        tiling: ArrayAttributeRef<'c, 't>,
        warp_dims: ArrayAttributeRef<'c, 't>,
        lane_dims: ArrayAttributeRef<'c, 't>,
        vector_dim: i32,
    ) -> Result<TiledLayoutAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_gpu()?)?;
        Ok(unsafe {
            TiledLayoutAttributeRef {
                handle: mlirMosaicGpuTiledLayoutAttrGet(
                    *self.handle.borrow_mut(),
                    tiling.to_c_api(),
                    warp_dims.to_c_api(),
                    lane_dims.to_c_api(),
                    vector_dim,
                ),
                context: self,
            }
        })
    }

    /// Creates a Mosaic GPU tile transform attribute owned by this [`Context`].
    pub fn mosaic_gpu_tile_transform_attribute<'c>(
        &'c self,
        tiling: &[i32],
    ) -> Result<TileTransformAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_gpu()?)?;
        let mut tiling = tiling.to_vec();
        let dimension_count =
            i32::try_from(tiling.len()).map_err(|_| Error::invalid_argument("too many Mosaic GPU tile dimensions"))?;
        unsafe {
            TileTransformAttributeRef::from_c_api(
                mlirMosaicGpuTileTransformAttrGet(*self.handle.borrow_mut(), tiling.as_mut_ptr(), dimension_count),
                self,
            )
            .map_err(|_| Error::invalid_argument("invalid arguments to `Context::mosaic_gpu_tile_transform_attribute`"))
        }
    }

    /// Creates a Mosaic GPU transpose transform attribute owned by this [`Context`].
    pub fn mosaic_gpu_transpose_transform_attribute<'c>(
        &'c self,
        permutation: &[i32],
    ) -> Result<TransposeTransformAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_gpu()?)?;
        let mut permutation = permutation.to_vec();
        let dimension_count = i32::try_from(permutation.len())
            .map_err(|_| Error::invalid_argument("too many Mosaic GPU transpose dimensions"))?;
        unsafe {
            TransposeTransformAttributeRef::from_c_api(
                mlirMosaicGpuTransposeTransformAttrGet(
                    *self.handle.borrow_mut(),
                    permutation.as_mut_ptr(),
                    dimension_count,
                ),
                self,
            )
            .map_err(|_| {
                Error::invalid_argument("invalid arguments to `Context::mosaic_gpu_transpose_transform_attribute`")
            })
        }
    }

    /// Creates a Mosaic GPU swizzle transform attribute owned by this [`Context`].
    pub fn mosaic_gpu_swizzle_transform_attribute<'c>(
        &'c self,
        swizzle: SwizzlingMode,
    ) -> Result<SwizzleTransformAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_gpu()?)?;
        Ok(unsafe {
            SwizzleTransformAttributeRef {
                handle: mlirMosaicGpuSwizzleTransformAttrGet(*self.handle.borrow_mut(), swizzle.as_i32()),
                context: self,
            }
        })
    }

    /// Creates a Mosaic GPU replicated copy partition attribute owned by this [`Context`].
    pub fn mosaic_gpu_copy_replicated_attribute<'c>(&'c self) -> Result<CopyReplicatedAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_gpu()?)?;
        Ok(unsafe {
            CopyReplicatedAttributeRef {
                handle: mlirMosaicGpuCopyReplicatedAttrGet(*self.handle.borrow_mut()),
                context: self,
            }
        })
    }

    /// Creates a Mosaic GPU partitioned copy partition attribute owned by this [`Context`].
    pub fn mosaic_gpu_copy_partitioned_attribute<'c>(
        &'c self,
        axis: i32,
    ) -> Result<CopyPartitionedAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_gpu()?)?;
        Ok(unsafe {
            CopyPartitionedAttributeRef {
                handle: mlirMosaicGpuCopyPartitionedAttrGet(*self.handle.borrow_mut(), axis),
                context: self,
            }
        })
    }

    /// Creates a Mosaic GPU tensor-memory address-space attribute owned by this [`Context`].
    pub fn mosaic_gpu_tmem_attribute<'c>(&'c self) -> Result<TmemAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_gpu()?)?;
        Ok(unsafe { TmemAttributeRef { handle: mlirMosaicGpuTmemAttrGet(*self.handle.borrow_mut()), context: self } })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::test_attribute_casting;

    use super::*;

    #[test]
    fn test_wg_strided_frag_layout_attribute() {
        let context = Context::new();
        let shape = context.dense_i64_array_attribute(&[2, 4]).unwrap();
        let attribute = context.mosaic_gpu_wg_strided_frag_layout_attribute(shape, 2).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.shape().unwrap(), shape);
        assert_eq!(attribute.vector_size(), 2);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_wg_splat_frag_layout_attribute() {
        let context = Context::new();
        let shape = context.dense_i64_array_attribute(&[2, 4]).unwrap();
        let attribute = context.mosaic_gpu_wg_splat_frag_layout_attribute(shape).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.shape().unwrap(), shape);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_replicated_attribute() {
        let context = Context::new();
        let attribute = context.mosaic_gpu_replicated_attribute(4).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.times(), 4);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_tiled_layout_attribute() {
        let context = Context::new();
        let tiling = context.array_attribute(&[context.mosaic_gpu_replicated_attribute(2).unwrap().as_ref()]);
        let element_type = context.signless_integer_type(64);
        let warp_dim_attributes =
            [context.integer_attribute(element_type, 0), context.integer_attribute(element_type, 1)];
        let warp_dims = context.array_attribute(&warp_dim_attributes);
        let lane_dim_attributes =
            [context.integer_attribute(element_type, 1), context.integer_attribute(element_type, 0)];
        let lane_dims = context.array_attribute(&lane_dim_attributes);
        let attribute = context.mosaic_gpu_tiled_layout_attribute(tiling, warp_dims, lane_dims, 1).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.tiling().unwrap(), tiling);
        assert_eq!(attribute.warp_dims().unwrap(), warp_dims);
        assert_eq!(attribute.lane_dims().unwrap(), lane_dims);
        assert_eq!(attribute.vector_dim(), 1);
        test_attribute_casting(attribute);
    }

    macro_rules! enum_attribute_tests {
        (
            $name:ident,
            $context_method:ident,
            $enum_name:ident,
            $attribute_name:ident,
            $value_1:ident,
            $value_2:ident $(,)?
        ) => {
            #[test]
            fn $name() {
                let context = Context::new();
                let attribute = context.$context_method($enum_name::$value_1).unwrap();
                assert_eq!(&context, attribute.context());
                assert_eq!(attribute.value().unwrap(), $enum_name::$value_1);
                assert_eq!($enum_name::from_str($enum_name::$value_1.as_str()), Some($enum_name::$value_1));
                assert_eq!($enum_name::from_i32($enum_name::$value_1.as_i32()), Some($enum_name::$value_1));
                assert_eq!($enum_name::from_str("invalid"), None);
                assert_eq!($enum_name::from_i32(-1), None);
                test_attribute_casting(attribute);

                let attribute_1 = context.$context_method($enum_name::$value_1).unwrap();
                let attribute_2 = context.$context_method($enum_name::$value_1).unwrap();
                assert_eq!(attribute_1, attribute_2);

                let attribute_2 = context.$context_method($enum_name::$value_2).unwrap();
                assert_ne!(attribute_1, attribute_2);

                let context = Context::new();
                let attribute_2 = context.$context_method($enum_name::$value_1).unwrap();
                assert_ne!(attribute_1, attribute_2);
            }
        };
    }

    enum_attribute_tests!(
        test_dimension_attribute,
        mosaic_gpu_dimension_attribute,
        Dimension,
        DimensionAttributeRef,
        X,
        Y,
    );

    enum_attribute_tests!(
        test_swizzling_mode_attribute,
        mosaic_gpu_swizzling_mode_attribute,
        SwizzlingMode,
        SwizzlingModeAttributeRef,
        NoSwizzle,
        Swizzle32Byte,
    );

    enum_attribute_tests!(
        test_tma_reduction_attribute,
        mosaic_gpu_tma_reduction_attribute,
        TmaReduction,
        TmaReductionAttributeRef,
        Add,
        Min,
    );

    enum_attribute_tests!(
        test_oob_fill_mode_attribute,
        mosaic_gpu_oob_fill_mode_attribute,
        OobFillMode,
        OobFillModeAttributeRef,
        Undefined,
        Zeros,
    );

    enum_attribute_tests!(
        test_multimem_load_reduction_type_attribute,
        mosaic_gpu_multimem_load_reduction_type_attribute,
        MultimemLoadReductionType,
        MultimemLoadReductionTypeAttributeRef,
        Add,
        Min,
    );

    enum_attribute_tests!(
        test_atomic_op_type_attribute,
        mosaic_gpu_atomic_op_type_attribute,
        AtomicOpType,
        AtomicOpTypeAttributeRef,
        Add,
        Min,
    );

    #[test]
    fn test_tile_transform_attribute() {
        let context = Context::new();
        let tiling = context.dense_i32_array_attribute(&[64, 32]).unwrap();
        let attribute = context.mosaic_gpu_tile_transform_attribute(&[64, 32]).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.tiling().unwrap(), tiling);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_transpose_transform_attribute() {
        let context = Context::new();
        let permutation = context.dense_i32_array_attribute(&[1, 0]).unwrap();
        let attribute = context.mosaic_gpu_transpose_transform_attribute(&[1, 0]).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.permutation().unwrap(), permutation);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_swizzle_transform_attribute() {
        let context = Context::new();
        let attribute = context.mosaic_gpu_swizzle_transform_attribute(SwizzlingMode::Swizzle64Byte).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.swizzle().unwrap(), SwizzlingMode::Swizzle64Byte);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_copy_replicated_attribute() {
        let context = Context::new();
        let attribute = context.mosaic_gpu_copy_replicated_attribute().unwrap();
        assert_eq!(&context, attribute.context());
        assert!(attribute.is::<CopyPartitionAttributeRef>());
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_copy_partitioned_attribute() {
        let context = Context::new();
        let attribute = context.mosaic_gpu_copy_partitioned_attribute(1).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.axis(), 1);
        assert!(attribute.is::<CopyPartitionAttributeRef>());
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_tmem_attribute() {
        let context = Context::new();
        let attribute = context.mosaic_gpu_tmem_attribute().unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.dialect().unwrap().namespace().unwrap(), "mosaic_gpu");
        test_attribute_casting(attribute);
    }
}
