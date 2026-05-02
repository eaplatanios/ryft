use ryft_xla_sys::bindings::{
    MlirAffineMap, MlirAttribute, MlirSparseTensorLevelFormat,
    MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_BATCH,
    MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED,
    MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_DENSE,
    MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED,
    MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M,
    MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_SINGLETON, MlirSparseTensorLevelPropertyNondefault,
    MlirSparseTensorLevelPropertyNondefault_MLIR_SPARSE_PROPERTY_NON_ORDERED,
    MlirSparseTensorLevelPropertyNondefault_MLIR_SPARSE_PROPERTY_NON_UNIQUE,
    MlirSparseTensorLevelPropertyNondefault_MLIR_SPARSE_PROPERTY_SOA, MlirSparseTensorLevelType,
    mlirAttributeIsASparseTensorEncodingAttr, mlirSparseTensorEncodingAttrBuildLvlType,
    mlirSparseTensorEncodingAttrGet, mlirSparseTensorEncodingAttrGetCrdWidth, mlirSparseTensorEncodingAttrGetDimToLvl,
    mlirSparseTensorEncodingAttrGetExplicitVal, mlirSparseTensorEncodingAttrGetImplicitVal,
    mlirSparseTensorEncodingAttrGetLvlFmt, mlirSparseTensorEncodingAttrGetLvlToDim,
    mlirSparseTensorEncodingAttrGetLvlType, mlirSparseTensorEncodingAttrGetPosWidth,
    mlirSparseTensorEncodingAttrGetStructuredM, mlirSparseTensorEncodingAttrGetStructuredN,
    mlirSparseTensorEncodingGetLvlRank,
};
use ryft_xla_sys::mlir::dialects::sparse_tensor::{
    MlirSparseTensorEnumAttribute, mlirAttributeIsASparseTensorDimSliceAttr, mlirAttributeIsASparseTensorEnumAttr,
    mlirSparseTensorDimSliceAttrGet, mlirSparseTensorDimSliceAttrGetOffset, mlirSparseTensorDimSliceAttrGetSize,
    mlirSparseTensorDimSliceAttrGetStride, mlirSparseTensorEncodingAttrGetDimSlice,
    mlirSparseTensorEncodingAttrGetDimSliceCount, mlirSparseTensorEncodingAttrGetWithDimSlices,
    mlirSparseTensorEnumAttrGet, mlirSparseTensorEnumAttrGetValue,
};

use crate::{AffineMap, Attribute, AttributeRef, Context, DialectHandle, FromWithContext, mlir_subtype_trait_impls};

/// Sparse tensor level format used by [`SparseTensorEncodingAttributeRef`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LevelFormat {
    /// Dense level storing every entry along the level.
    Dense,

    /// Batch level storing every entry but not linearizing it with adjacent levels.
    Batch,

    /// Compressed level storing only coordinates for explicit entries.
    Compressed,

    /// Singleton level where each coordinate has no siblings.
    Singleton,

    /// Loose compressed level with free space between position intervals.
    LooseCompressed,

    /// Structured n-out-of-m level format.
    NOutOfM,
}

impl LevelFormat {
    /// Returns the MLIR C API representation of this level format.
    pub fn to_c_api(&self) -> MlirSparseTensorLevelFormat {
        match self {
            Self::Dense => MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_DENSE,
            Self::Batch => MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_BATCH,
            Self::Compressed => MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED,
            Self::Singleton => MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_SINGLETON,
            Self::LooseCompressed => MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED,
            Self::NOutOfM => MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M,
        }
    }

    /// Constructs a [`LevelFormat`] from the MLIR C API representation.
    #[allow(non_upper_case_globals)]
    pub fn from_c_api(value: MlirSparseTensorLevelFormat) -> Option<Self> {
        match value {
            MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_DENSE => Some(Self::Dense),
            MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_BATCH => Some(Self::Batch),
            MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED => Some(Self::Compressed),
            MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_SINGLETON => Some(Self::Singleton),
            MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED => Some(Self::LooseCompressed),
            MlirSparseTensorLevelFormat_MLIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M => Some(Self::NOutOfM),
            _ => None,
        }
    }
}

/// Non-default sparse tensor level property.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LevelProperty {
    /// Coordinates may contain duplicates at the level.
    NonUnique,

    /// Coordinates may appear in an arbitrary order at the level.
    NonOrdered,

    /// Singleton coordinates use a structure-of-arrays layout.
    StructureOfArrays,
}

impl LevelProperty {
    /// Returns the MLIR C API representation of this level property.
    pub fn to_c_api(&self) -> MlirSparseTensorLevelPropertyNondefault {
        match self {
            Self::NonUnique => MlirSparseTensorLevelPropertyNondefault_MLIR_SPARSE_PROPERTY_NON_UNIQUE,
            Self::NonOrdered => MlirSparseTensorLevelPropertyNondefault_MLIR_SPARSE_PROPERTY_NON_ORDERED,
            Self::StructureOfArrays => MlirSparseTensorLevelPropertyNondefault_MLIR_SPARSE_PROPERTY_SOA,
        }
    }
}

/// Sparse tensor level type combining a storage format, applicable non-default properties, and structured parameters.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LevelType {
    /// Encoded MLIR level-type bits.
    bits: MlirSparseTensorLevelType,
}

impl LevelType {
    /// Constructs a level type from the provided format and non-default properties.
    pub fn new(format: LevelFormat, properties: &[LevelProperty]) -> Self {
        Self::structured(format, properties, 0, 0)
    }

    /// Constructs a structured n-out-of-m level type.
    pub fn n_out_of_m(n: u32, m: u32) -> Self {
        Self::structured(LevelFormat::NOutOfM, &[], n, m)
    }

    /// Constructs a level type from raw MLIR level-type bits.
    pub fn from_c_api(bits: MlirSparseTensorLevelType) -> Self {
        Self { bits }
    }

    /// Returns the raw MLIR level-type bits.
    pub fn to_c_api(&self) -> MlirSparseTensorLevelType {
        self.bits
    }

    /// Returns the level format.
    pub fn format(&self) -> LevelFormat {
        LevelFormat::from_c_api((self.bits & 0xffff0000) as MlirSparseTensorLevelFormat)
            .expect("invalid sparse tensor level format")
    }

    /// Returns `true` if this level type has the provided non-default property.
    pub fn has_property(&self, property: LevelProperty) -> bool {
        self.bits & property.to_c_api() as u64 != 0
    }

    /// Returns the `n` value for structured n-out-of-m level types.
    pub fn structured_n(&self) -> Option<u32> {
        if self.format() == LevelFormat::NOutOfM {
            Some(unsafe { mlirSparseTensorEncodingAttrGetStructuredN(self.bits) })
        } else {
            None
        }
    }

    /// Returns the `m` value for structured n-out-of-m level types.
    pub fn structured_m(&self) -> Option<u32> {
        if self.format() == LevelFormat::NOutOfM {
            Some(unsafe { mlirSparseTensorEncodingAttrGetStructuredM(self.bits) })
        } else {
            None
        }
    }

    /// Constructs a level type using the upstream MLIR level-type builder.
    fn structured(format: LevelFormat, properties: &[LevelProperty], n: u32, m: u32) -> Self {
        let properties = properties.iter().map(LevelProperty::to_c_api).collect::<Vec<_>>();
        Self {
            bits: unsafe {
                mlirSparseTensorEncodingAttrBuildLvlType(
                    format.to_c_api(),
                    properties.as_ptr(),
                    properties.len() as u32,
                    n,
                    m,
                )
            },
        }
    }
}

impl From<LevelFormat> for LevelType {
    fn from(value: LevelFormat) -> Self {
        Self::new(value, &[])
    }
}

/// Sparse tensor dimension-slice [`Attribute`].
#[derive(Copy, Clone)]
pub struct SparseTensorDimSliceAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl SparseTensorDimSliceAttributeRef<'_, '_> {
    /// Sentinel value used by MLIR for dynamic offsets, sizes, and strides.
    pub const DYNAMIC: i64 = -1;

    /// Returns the slice offset, or [`SparseTensorDimSliceAttributeRef::DYNAMIC`] when it is dynamic.
    pub fn offset(&self) -> i64 {
        unsafe { mlirSparseTensorDimSliceAttrGetOffset(self.handle) }
    }

    /// Returns the slice size, or [`SparseTensorDimSliceAttributeRef::DYNAMIC`] when it is dynamic.
    pub fn size(&self) -> i64 {
        unsafe { mlirSparseTensorDimSliceAttrGetSize(self.handle) }
    }

    /// Returns the slice stride, or [`SparseTensorDimSliceAttributeRef::DYNAMIC`] when it is dynamic.
    pub fn stride(&self) -> i64 {
        unsafe { mlirSparseTensorDimSliceAttrGetStride(self.handle) }
    }
}

impl<'c, 't> Attribute<'c, 't> for SparseTensorDimSliceAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsASparseTensorDimSliceAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(SparseTensorDimSliceAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Sparse tensor encoding [`Attribute`] that describes the sparse storage format of a tensor type.
#[derive(Copy, Clone)]
pub struct SparseTensorEncodingAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> SparseTensorEncodingAttributeRef<'c, 't> {
    /// Returns the number of storage levels in this encoding.
    pub fn level_rank(&self) -> usize {
        unsafe { mlirSparseTensorEncodingGetLvlRank(self.handle).cast_unsigned() }
    }

    /// Returns the level types in storage-level order.
    pub fn level_types(&self) -> Vec<LevelType> {
        (0..self.level_rank()).map(|level| self.level_type(level)).collect()
    }

    /// Returns the level type for `level`.
    pub fn level_type(&self, level: usize) -> LevelType {
        LevelType::from_c_api(unsafe { mlirSparseTensorEncodingAttrGetLvlType(self.handle, level.cast_signed()) })
    }

    /// Returns the level format for `level`.
    pub fn level_format(&self, level: usize) -> LevelFormat {
        LevelFormat::from_c_api(unsafe { mlirSparseTensorEncodingAttrGetLvlFmt(self.handle, level.cast_signed()) })
            .expect("invalid sparse tensor level format")
    }

    /// Returns the dimension-to-level affine map.
    pub fn dimension_to_level(&self) -> AffineMap<'c, 't> {
        unsafe {
            AffineMap::from_c_api(mlirSparseTensorEncodingAttrGetDimToLvl(self.handle), self.context)
                .expect("invalid sparse tensor dimension-to-level map")
        }
    }

    /// Returns the level-to-dimension affine map.
    pub fn level_to_dimension(&self) -> AffineMap<'c, 't> {
        unsafe {
            AffineMap::from_c_api(mlirSparseTensorEncodingAttrGetLvlToDim(self.handle), self.context)
                .expect("invalid sparse tensor level-to-dimension map")
        }
    }

    /// Returns the position-overhead bit width, or zero for the target-native index width.
    pub fn position_width(&self) -> u32 {
        unsafe { mlirSparseTensorEncodingAttrGetPosWidth(self.handle) as u32 }
    }

    /// Returns the coordinate-overhead bit width, or zero for the target-native index width.
    pub fn coordinate_width(&self) -> u32 {
        unsafe { mlirSparseTensorEncodingAttrGetCrdWidth(self.handle) as u32 }
    }

    /// Returns the optional explicit value used by binary-valued sparse tensors.
    pub fn explicit_value(&self) -> Option<AttributeRef<'c, 't>> {
        unsafe { AttributeRef::from_c_api(mlirSparseTensorEncodingAttrGetExplicitVal(self.handle), self.context) }
    }

    /// Returns the optional implicit value for unstored tensor entries.
    pub fn implicit_value(&self) -> Option<AttributeRef<'c, 't>> {
        unsafe { AttributeRef::from_c_api(mlirSparseTensorEncodingAttrGetImplicitVal(self.handle), self.context) }
    }

    /// Returns the dimension-slice metadata in dimension order.
    pub fn dimension_slices(&self) -> Vec<SparseTensorDimSliceAttributeRef<'c, 't>> {
        let count = unsafe { mlirSparseTensorEncodingAttrGetDimSliceCount(self.handle).cast_unsigned() };
        (0..count)
            .map(|dimension| unsafe {
                SparseTensorDimSliceAttributeRef::from_c_api(
                    mlirSparseTensorEncodingAttrGetDimSlice(self.handle, dimension.cast_signed()),
                    self.context,
                )
                .expect("invalid sparse tensor dimension slice")
            })
            .collect()
    }
}

impl<'c, 't> Attribute<'c, 't> for SparseTensorEncodingAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsASparseTensorEncodingAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(SparseTensorEncodingAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Sparse tensor storage-specifier field kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum StorageSpecifierKind {
    /// Level-size metadata field.
    LevelSize,

    /// Position-buffer size metadata field.
    PositionMemorySize,

    /// Coordinate-buffer size metadata field.
    CoordinateMemorySize,

    /// Value-buffer size metadata field.
    ValueMemorySize,

    /// Dimension-slice offset metadata field.
    DimensionOffset,

    /// Dimension-slice stride metadata field.
    DimensionStride,
}

impl StorageSpecifierKind {
    /// Returns the integer value used by MLIR for this kind.
    pub fn value(&self) -> u32 {
        match self {
            Self::LevelSize => 0,
            Self::PositionMemorySize => 1,
            Self::CoordinateMemorySize => 2,
            Self::ValueMemorySize => 3,
            Self::DimensionOffset => 4,
            Self::DimensionStride => 5,
        }
    }

    /// Constructs a [`StorageSpecifierKind`] from the integer value used by MLIR.
    pub fn from_value(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::LevelSize),
            1 => Some(Self::PositionMemorySize),
            2 => Some(Self::CoordinateMemorySize),
            3 => Some(Self::ValueMemorySize),
            4 => Some(Self::DimensionOffset),
            5 => Some(Self::DimensionStride),
            _ => None,
        }
    }

    /// Returns the textual MLIR spelling for this kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LevelSize => "lvl_sz",
            Self::PositionMemorySize => "pos_mem_sz",
            Self::CoordinateMemorySize => "crd_mem_sz",
            Self::ValueMemorySize => "val_mem_sz",
            Self::DimensionOffset => "dim_offset",
            Self::DimensionStride => "dim_stride",
        }
    }
}

/// MLIR sparse tensor storage-specifier kind [`Attribute`].
#[derive(Copy, Clone)]
pub struct StorageSpecifierKindAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl StorageSpecifierKindAttributeRef<'_, '_> {
    /// Returns the storage-specifier kind stored in this attribute.
    pub fn value(&self) -> StorageSpecifierKind {
        StorageSpecifierKind::from_value(unsafe {
            mlirSparseTensorEnumAttrGetValue(
                self.handle,
                MlirSparseTensorEnumAttribute::MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_STORAGE_SPECIFIER_KIND,
            )
        })
        .expect("invalid sparse tensor storage specifier kind")
    }
}

impl<'c, 't> Attribute<'c, 't> for StorageSpecifierKindAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null()
            && unsafe {
                mlirAttributeIsASparseTensorEnumAttr(
                    handle,
                    MlirSparseTensorEnumAttribute::MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_STORAGE_SPECIFIER_KIND,
                )
            }
        {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(StorageSpecifierKindAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'c, 't> FromWithContext<'c, 't, StorageSpecifierKind> for StorageSpecifierKindAttributeRef<'c, 't> {
    fn from_with_context(value: StorageSpecifierKind, context: &'c Context<'t>) -> Self {
        context.sparse_tensor_storage_specifier_kind_attribute(value)
    }
}

/// Sparse tensor sorting algorithm.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SortKind {
    /// Hybrid quicksort algorithm.
    HybridQuickSort,

    /// Stable insertion sort algorithm.
    InsertionSortStable,

    /// Quicksort algorithm.
    QuickSort,

    /// Heap sort algorithm.
    HeapSort,
}

impl SortKind {
    /// Returns the integer value used by MLIR for this sort kind.
    pub fn value(&self) -> u32 {
        match self {
            Self::HybridQuickSort => 0,
            Self::InsertionSortStable => 1,
            Self::QuickSort => 2,
            Self::HeapSort => 3,
        }
    }

    /// Constructs a [`SortKind`] from the integer value used by MLIR.
    pub fn from_value(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::HybridQuickSort),
            1 => Some(Self::InsertionSortStable),
            2 => Some(Self::QuickSort),
            3 => Some(Self::HeapSort),
            _ => None,
        }
    }

    /// Returns the textual MLIR spelling for this sort kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HybridQuickSort => "hybrid_quick_sort",
            Self::InsertionSortStable => "insertion_sort_stable",
            Self::QuickSort => "quick_sort",
            Self::HeapSort => "heap_sort",
        }
    }
}

/// MLIR sparse tensor sort-kind [`Attribute`].
#[derive(Copy, Clone)]
pub struct SortKindAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl SortKindAttributeRef<'_, '_> {
    /// Returns the sort kind stored in this attribute.
    pub fn value(&self) -> SortKind {
        SortKind::from_value(unsafe {
            mlirSparseTensorEnumAttrGetValue(
                self.handle,
                MlirSparseTensorEnumAttribute::MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_SORT_KIND,
            )
        })
        .expect("invalid sparse tensor sort kind")
    }
}

impl<'c, 't> Attribute<'c, 't> for SortKindAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null()
            && unsafe {
                mlirAttributeIsASparseTensorEnumAttr(
                    handle,
                    MlirSparseTensorEnumAttribute::MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_SORT_KIND,
                )
            }
        {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(SortKindAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'c, 't> FromWithContext<'c, 't, SortKind> for SortKindAttributeRef<'c, 't> {
    fn from_with_context(value: SortKind, context: &'c Context<'t>) -> Self {
        context.sparse_tensor_sort_kind_attribute(value)
    }
}

/// Sparse tensor coordinate translation direction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CoordinateTranslationDirection {
    /// Translate dimension coordinates to level coordinates.
    DimensionToLevel,

    /// Translate level coordinates to dimension coordinates.
    LevelToDimension,
}

impl CoordinateTranslationDirection {
    /// Returns the integer value used by MLIR for this direction.
    pub fn value(&self) -> u32 {
        match self {
            Self::DimensionToLevel => 0,
            Self::LevelToDimension => 1,
        }
    }

    /// Constructs a [`CoordinateTranslationDirection`] from the integer value used by MLIR.
    pub fn from_value(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::DimensionToLevel),
            1 => Some(Self::LevelToDimension),
            _ => None,
        }
    }

    /// Returns the textual MLIR spelling for this direction.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DimensionToLevel => "dim_to_lvl",
            Self::LevelToDimension => "lvl_to_dim",
        }
    }
}

/// MLIR sparse tensor coordinate-translation direction [`Attribute`].
#[derive(Copy, Clone)]
pub struct CoordinateTranslationDirectionAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl CoordinateTranslationDirectionAttributeRef<'_, '_> {
    /// Returns the coordinate-translation direction stored in this attribute.
    pub fn value(&self) -> CoordinateTranslationDirection {
        CoordinateTranslationDirection::from_value(unsafe {
            mlirSparseTensorEnumAttrGetValue(
                self.handle,
                MlirSparseTensorEnumAttribute::MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_CRD_TRANS_DIRECTION,
            )
        })
        .expect("invalid sparse tensor coordinate translation direction")
    }
}

impl<'c, 't> Attribute<'c, 't> for CoordinateTranslationDirectionAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null()
            && unsafe {
                mlirAttributeIsASparseTensorEnumAttr(
                    handle,
                    MlirSparseTensorEnumAttribute::MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_CRD_TRANS_DIRECTION,
                )
            }
        {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(
    CoordinateTranslationDirectionAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
);

impl<'c, 't> FromWithContext<'c, 't, CoordinateTranslationDirection>
    for CoordinateTranslationDirectionAttributeRef<'c, 't>
{
    fn from_with_context(value: CoordinateTranslationDirection, context: &'c Context<'t>) -> Self {
        context.sparse_tensor_coordinate_translation_direction_attribute(value)
    }
}

impl<'t> Context<'t> {
    /// Creates a new sparse tensor dimension-slice attribute owned by this [`Context`].
    pub fn sparse_tensor_dim_slice_attribute<'c>(
        &'c self,
        offset: i64,
        size: i64,
        stride: i64,
    ) -> SparseTensorDimSliceAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::sparse_tensor());
        unsafe {
            SparseTensorDimSliceAttributeRef::from_c_api(
                mlirSparseTensorDimSliceAttrGet(*self.handle.borrow(), offset, size, stride),
                self,
            )
            .expect("invalid sparse tensor dimension slice attribute")
        }
    }

    /// Creates a new sparse tensor encoding attribute owned by this [`Context`].
    pub fn sparse_tensor_encoding_attribute<'c>(
        &'c self,
        level_types: &[LevelType],
        dimension_to_level: Option<AffineMap<'c, 't>>,
        level_to_dimension: Option<AffineMap<'c, 't>>,
        position_width: u32,
        coordinate_width: u32,
        explicit_value: Option<AttributeRef<'c, 't>>,
        implicit_value: Option<AttributeRef<'c, 't>>,
        dimension_slices: &[SparseTensorDimSliceAttributeRef<'c, 't>],
    ) -> SparseTensorEncodingAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::sparse_tensor());
        let level_types = level_types.iter().map(LevelType::to_c_api).collect::<Vec<_>>();
        let dimension_slices = dimension_slices
            .iter()
            .map(|dimension_slice| unsafe { dimension_slice.to_c_api() })
            .collect::<Vec<_>>();
        let null_affine_map = MlirAffineMap { ptr: std::ptr::null_mut() };
        let dimension_to_level = dimension_to_level.map(|map| unsafe { map.to_c_api() }).unwrap_or(null_affine_map);
        let level_to_dimension = level_to_dimension.map(|map| unsafe { map.to_c_api() }).unwrap_or(null_affine_map);
        let explicit_value = explicit_value.unwrap_or_else(|| self.null_attribute());
        let implicit_value = implicit_value.unwrap_or_else(|| self.null_attribute());
        let handle = if dimension_slices.is_empty() {
            unsafe {
                mlirSparseTensorEncodingAttrGet(
                    *self.handle.borrow(),
                    level_types.len().cast_signed(),
                    level_types.as_ptr(),
                    dimension_to_level,
                    level_to_dimension,
                    position_width as std::ffi::c_int,
                    coordinate_width as std::ffi::c_int,
                    explicit_value.to_c_api(),
                    implicit_value.to_c_api(),
                )
            }
        } else {
            unsafe {
                mlirSparseTensorEncodingAttrGetWithDimSlices(
                    *self.handle.borrow(),
                    level_types.len().cast_signed(),
                    level_types.as_ptr(),
                    dimension_to_level,
                    level_to_dimension,
                    position_width as std::ffi::c_int,
                    coordinate_width as std::ffi::c_int,
                    explicit_value.to_c_api(),
                    implicit_value.to_c_api(),
                    dimension_slices.len().cast_signed(),
                    dimension_slices.as_ptr(),
                )
            }
        };
        unsafe { SparseTensorEncodingAttributeRef::from_c_api(handle, self) }
            .expect("invalid sparse tensor encoding attribute")
    }

    /// Creates a new sparse tensor storage-specifier kind attribute owned by this [`Context`].
    pub fn sparse_tensor_storage_specifier_kind_attribute<'c>(
        &'c self,
        value: StorageSpecifierKind,
    ) -> StorageSpecifierKindAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::sparse_tensor());
        unsafe {
            StorageSpecifierKindAttributeRef::from_c_api(
                mlirSparseTensorEnumAttrGet(
                    *self.handle.borrow(),
                    MlirSparseTensorEnumAttribute::MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_STORAGE_SPECIFIER_KIND,
                    value.value(),
                ),
                self,
            )
            .expect("invalid sparse tensor storage specifier kind attribute")
        }
    }

    /// Creates a new sparse tensor sort-kind attribute owned by this [`Context`].
    pub fn sparse_tensor_sort_kind_attribute<'c>(&'c self, value: SortKind) -> SortKindAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::sparse_tensor());
        unsafe {
            SortKindAttributeRef::from_c_api(
                mlirSparseTensorEnumAttrGet(
                    *self.handle.borrow(),
                    MlirSparseTensorEnumAttribute::MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_SORT_KIND,
                    value.value(),
                ),
                self,
            )
            .expect("invalid sparse tensor sort kind attribute")
        }
    }

    /// Creates a new sparse tensor coordinate-translation direction attribute owned by this [`Context`].
    pub fn sparse_tensor_coordinate_translation_direction_attribute<'c>(
        &'c self,
        value: CoordinateTranslationDirection,
    ) -> CoordinateTranslationDirectionAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::sparse_tensor());
        unsafe {
            CoordinateTranslationDirectionAttributeRef::from_c_api(
                mlirSparseTensorEnumAttrGet(
                    *self.handle.borrow(),
                    MlirSparseTensorEnumAttribute::MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_CRD_TRANS_DIRECTION,
                    value.value(),
                ),
                self,
            )
            .expect("invalid sparse tensor coordinate translation direction attribute")
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};
    use crate::{Attribute, Context};

    use super::*;

    #[test]
    fn test_level_type() {
        let level_type = LevelType::new(LevelFormat::Compressed, &[LevelProperty::NonUnique]);
        assert_eq!(level_type.format(), LevelFormat::Compressed);
        assert!(level_type.has_property(LevelProperty::NonUnique));
        assert!(!level_type.has_property(LevelProperty::NonOrdered));
        assert_eq!(LevelType::from_c_api(level_type.to_c_api()), level_type);
        assert_eq!(LevelType::from(LevelFormat::Dense), LevelType::new(LevelFormat::Dense, &[]));
    }

    #[test]
    fn test_level_type_structured() {
        let level_type = LevelType::n_out_of_m(2, 4);
        assert_eq!(level_type.format(), LevelFormat::NOutOfM);
        assert_eq!(level_type.structured_n(), Some(2));
        assert_eq!(level_type.structured_m(), Some(4));
    }

    #[test]
    fn test_dim_slice_attribute() {
        let context = Context::new();
        let attribute = context.sparse_tensor_dim_slice_attribute(1, SparseTensorDimSliceAttributeRef::DYNAMIC, 2);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.offset(), 1);
        assert_eq!(attribute.size(), SparseTensorDimSliceAttributeRef::DYNAMIC);
        assert_eq!(attribute.stride(), 2);
    }

    #[test]
    fn test_dim_slice_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.sparse_tensor_dim_slice_attribute(1, 4, 2);
        let attribute_2 = context.sparse_tensor_dim_slice_attribute(1, 4, 2);
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.sparse_tensor_dim_slice_attribute(1, 4, 1);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.sparse_tensor_dim_slice_attribute(1, 4, 2);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_dim_slice_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.sparse_tensor_dim_slice_attribute(1, SparseTensorDimSliceAttributeRef::DYNAMIC, 2);
        test_attribute_display_and_debug(attribute, "#sparse_tensor<slice(1, ?, 2)>");
    }

    #[test]
    fn test_dim_slice_attribute_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::sparse_tensor());
        let attribute = context.sparse_tensor_dim_slice_attribute(1, SparseTensorDimSliceAttributeRef::DYNAMIC, 2);
        assert_eq!(
            context
                .parse_attribute("#sparse_tensor<slice(1, ?, 2)>")
                .unwrap()
                .cast::<SparseTensorDimSliceAttributeRef>()
                .unwrap(),
            attribute,
        );
    }

    #[test]
    fn test_dim_slice_attribute_casting() {
        let context = Context::new();
        let attribute = context.sparse_tensor_dim_slice_attribute(1, SparseTensorDimSliceAttributeRef::DYNAMIC, 2);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_encoding_attribute() {
        let context = Context::new();
        let slice = context.sparse_tensor_dim_slice_attribute(1, SparseTensorDimSliceAttributeRef::DYNAMIC, 2);
        let dimension_to_level = context.identity_affine_map(1);
        let attribute = context.sparse_tensor_encoding_attribute(
            &[LevelType::from(LevelFormat::Compressed)],
            Some(dimension_to_level),
            None,
            32,
            64,
            None,
            None,
            &[slice],
        );
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.level_rank(), 1);
        assert_eq!(attribute.level_types(), vec![LevelType::from(LevelFormat::Compressed)]);
        assert_eq!(attribute.level_type(0), LevelType::from(LevelFormat::Compressed));
        assert_eq!(attribute.level_format(0), LevelFormat::Compressed);
        assert_eq!(attribute.dimension_to_level(), dimension_to_level);
        assert_eq!(attribute.position_width(), 32);
        assert_eq!(attribute.coordinate_width(), 64);
        assert_eq!(attribute.explicit_value(), None);
        assert_eq!(attribute.implicit_value(), None);
        assert_eq!(attribute.dimension_slices(), vec![slice]);
    }

    #[test]
    fn test_encoding_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.sparse_tensor_encoding_attribute(
            &[LevelType::from(LevelFormat::Compressed)],
            Some(context.identity_affine_map(1)),
            None,
            32,
            64,
            None,
            None,
            &[],
        );
        let attribute_2 = context.sparse_tensor_encoding_attribute(
            &[LevelType::from(LevelFormat::Compressed)],
            Some(context.identity_affine_map(1)),
            None,
            32,
            64,
            None,
            None,
            &[],
        );
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.sparse_tensor_encoding_attribute(
            &[LevelType::from(LevelFormat::Dense)],
            Some(context.identity_affine_map(1)),
            None,
            32,
            64,
            None,
            None,
            &[],
        );
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.sparse_tensor_encoding_attribute(
            &[LevelType::from(LevelFormat::Compressed)],
            Some(context.identity_affine_map(1)),
            None,
            32,
            64,
            None,
            None,
            &[],
        );
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_encoding_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.sparse_tensor_encoding_attribute(
            &[LevelType::from(LevelFormat::Compressed)],
            Some(context.identity_affine_map(1)),
            None,
            32,
            64,
            None,
            None,
            &[],
        );
        test_attribute_display_and_debug(
            attribute,
            "#sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed), posWidth = 32, crdWidth = 64 }>",
        );
    }

    #[test]
    fn test_encoding_attribute_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::sparse_tensor());
        let attribute = context.sparse_tensor_encoding_attribute(
            &[LevelType::from(LevelFormat::Compressed)],
            Some(context.identity_affine_map(1)),
            None,
            32,
            64,
            None,
            None,
            &[],
        );
        assert_eq!(
            context
                .parse_attribute(
                    "#sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed), posWidth = 32, crdWidth = 64 }>",
                )
                .unwrap()
                .cast::<SparseTensorEncodingAttributeRef>()
                .unwrap(),
            attribute,
        );
    }

    #[test]
    fn test_encoding_attribute_casting() {
        let context = Context::new();
        let attribute = context.sparse_tensor_encoding_attribute(
            &[LevelType::from(LevelFormat::Compressed)],
            Some(context.identity_affine_map(1)),
            None,
            32,
            64,
            None,
            None,
            &[],
        );
        test_attribute_casting(attribute);
    }

    macro_rules! sparse_tensor_enum_attribute_tests {
        (
            $test_name:ident,
            $constructor:ident,
            $enum_name:ident,
            $attribute_name:ident,
            $first:ident,
            $second:ident,
            $expected_prefix:literal
            $(,)?
        ) => {
            paste::paste! {
                #[test]
                fn [<test_ $test_name _attribute>]() {
                    let context = Context::new();
                    let attribute = context.$constructor($enum_name::$first);
                    assert_eq!(&context, attribute.context());
                    assert_eq!(attribute.value(), $enum_name::$first);
                    assert_eq!($enum_name::from_value(attribute.value().value()), Some($enum_name::$first));
                }

                #[test]
                fn [<test_ $test_name _attribute_equality>]() {
                    let context = Context::new();
                    let attribute_1 = context.$constructor($enum_name::$first);
                    let attribute_2 = context.$constructor($enum_name::$first);
                    assert_eq!(attribute_1, attribute_2);

                    let attribute_2 = context.$constructor($enum_name::$second);
                    assert_ne!(attribute_1, attribute_2);

                    let context = Context::new();
                    let attribute_2 = context.$constructor($enum_name::$first);
                    assert_ne!(attribute_1, attribute_2);
                }

                #[test]
                fn [<test_ $test_name _attribute_display_and_debug>]() {
                    let context = Context::new();
                    let attribute = context.$constructor($enum_name::$first);
                    let expected = format!("{}{}>", $expected_prefix, $enum_name::$first.as_str());
                    test_attribute_display_and_debug(attribute, Box::leak(expected.into_boxed_str()));
                }

                #[test]
                fn [<test_ $test_name _attribute_parsing>]() {
                    let context = Context::new();
                    context.load_dialect(DialectHandle::sparse_tensor());
                    let attribute = context.$constructor($enum_name::$first);
                    let source = format!("{}{}>", $expected_prefix, $enum_name::$first.as_str());
                    assert_eq!(
                        context.parse_attribute(&source).unwrap().cast::<$attribute_name>().unwrap(),
                        attribute,
                    );
                }

                #[test]
                fn [<test_ $test_name _attribute_casting>]() {
                    let context = Context::new();
                    let attribute = context.$constructor($enum_name::$first);
                    test_attribute_casting(attribute);
                }
            }
        };
    }

    sparse_tensor_enum_attribute_tests!(
        storage_specifier_kind,
        sparse_tensor_storage_specifier_kind_attribute,
        StorageSpecifierKind,
        StorageSpecifierKindAttributeRef,
        CoordinateMemorySize,
        ValueMemorySize,
        "#sparse_tensor.kind<",
    );

    sparse_tensor_enum_attribute_tests!(
        sort_kind,
        sparse_tensor_sort_kind_attribute,
        SortKind,
        SortKindAttributeRef,
        QuickSort,
        HeapSort,
        "#sparse_tensor<SparseTensorSortAlgorithm ",
    );

    sparse_tensor_enum_attribute_tests!(
        coordinate_translation_direction,
        sparse_tensor_coordinate_translation_direction_attribute,
        CoordinateTranslationDirection,
        CoordinateTranslationDirectionAttributeRef,
        DimensionToLevel,
        LevelToDimension,
        "#sparse_tensor<CrdTransDirection ",
    );
}
