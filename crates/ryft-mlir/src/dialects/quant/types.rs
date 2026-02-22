use ryft_xla_sys::bindings::{
    MlirType, mlirAnyQuantizedTypeGet, mlirAnyQuantizedTypeGetTypeID, mlirCalibratedQuantizedTypeGet,
    mlirCalibratedQuantizedTypeGetMax, mlirCalibratedQuantizedTypeGetMin, mlirCalibratedQuantizedTypeGetTypeID,
    mlirQuantizedTypeCastExpressedToStorageType, mlirQuantizedTypeCastFromExpressedType,
    mlirQuantizedTypeCastFromStorageType, mlirQuantizedTypeCastToExpressedType, mlirQuantizedTypeCastToStorageType,
    mlirQuantizedTypeGetDefaultMaximumForInteger, mlirQuantizedTypeGetDefaultMinimumForInteger,
    mlirQuantizedTypeGetExpressedType, mlirQuantizedTypeGetFlags, mlirQuantizedTypeGetQuantizedElementType,
    mlirQuantizedTypeGetSignedFlag, mlirQuantizedTypeGetStorageType, mlirQuantizedTypeGetStorageTypeIntegralWidth,
    mlirQuantizedTypeGetStorageTypeMax, mlirQuantizedTypeGetStorageTypeMin, mlirQuantizedTypeIsCompatibleExpressedType,
    mlirQuantizedTypeIsSigned, mlirUniformQuantizedPerAxisTypeGet, mlirUniformQuantizedPerAxisTypeGetNumDims,
    mlirUniformQuantizedPerAxisTypeGetQuantizedDimension, mlirUniformQuantizedPerAxisTypeGetScale,
    mlirUniformQuantizedPerAxisTypeGetTypeID, mlirUniformQuantizedPerAxisTypeGetZeroPoint,
    mlirUniformQuantizedPerAxisTypeIsFixedPoint, mlirUniformQuantizedSubChannelTypeGet,
    mlirUniformQuantizedSubChannelTypeGetBlockSize, mlirUniformQuantizedSubChannelTypeGetNumBlockSizes,
    mlirUniformQuantizedSubChannelTypeGetQuantizedDimension, mlirUniformQuantizedSubChannelTypeGetScales,
    mlirUniformQuantizedSubChannelTypeGetTypeID, mlirUniformQuantizedSubChannelTypeGetZeroPoints,
    mlirUniformQuantizedTypeGet, mlirUniformQuantizedTypeGetScale, mlirUniformQuantizedTypeGetTypeID,
    mlirUniformQuantizedTypeGetZeroPoint, mlirUniformQuantizedTypeIsFixedPoint,
};

use crate::{
    Attribute, AttributeRef, Context, DialectHandle, IntegerTypeRef, Type, TypeId, TypeRef, mlir_subtype_trait_impls,
};

/// Built-in MLIR [`Type`] that represents a _quantized_ type.
///
/// This `trait` acts effectively as the super-type of all MLIR quantized [`Type`]s and can be checked and
/// specialized using the [`QuantizedType::is`](TypeRef::is) and [`QuantizedType::cast`](TypeRef::cast) functions.
pub trait QuantizedType<'c, 't: 'c>: Type<'c, 't> {
    /// Returns the original expressed [`TypeRef`] that this [`QuantizedType`] approximates. Note that this presumes
    /// that the quantized type was always derived from a floating-point type, which in the broadest definition, is not
    /// necessarily correct. However, at a high level, no examples of such usage are presently known and the restriction
    /// serves some useful purposes (e.g., always being able to reverse a transformation or measure error). In most
    /// cases, this function will return [`Float32TypeRef`](crate::Float32TypeRef).
    fn expressed_type(&self) -> TypeRef<'c, 't> {
        unsafe { TypeRef::from_c_api(mlirQuantizedTypeGetExpressedType(self.to_c_api()), self.context()).unwrap() }
    }

    /// Returns the quantization flags of this [`QuantizedType`] as a `u32` whose bits correspond to the flags.
    fn flags(&self) -> u32 {
        unsafe { mlirQuantizedTypeGetFlags(self.to_c_api()) }
    }

    /// Returns the bit flag used by MLIR to indicate signed [`QuantizedType`]s.
    fn signed_flag() -> u32 {
        unsafe { mlirQuantizedTypeGetSignedFlag() }
    }

    /// Returns `true` if this [`QuantizedType`] is signed.
    fn is_signed(&self) -> bool {
        unsafe { mlirQuantizedTypeIsSigned(self.to_c_api()) }
    }

    /// Returns the minimum representable value for an integer storage type with the provided bit width and signedness.
    /// [`Self::storage_type_minimum`] must be greater than or equal to this value.
    fn default_minimum_for_integer(is_signed: bool, width: usize) -> i64 {
        unsafe { mlirQuantizedTypeGetDefaultMinimumForInteger(is_signed, width as u32) }
    }

    /// Returns the maximum representable value for an integer storage type with the provided bit width and signedness.
    /// [`Self::storage_type_maximum`] must be less than or equal to this value.
    fn default_maximum_for_integer(is_signed: bool, width: usize) -> i64 {
        unsafe { mlirQuantizedTypeGetDefaultMaximumForInteger(is_signed, width as u32) }
    }

    /// Returns the integer storage type of this [`QuantizedType`]. This is the integer type of the values stored in
    /// memory and conveys the bit width and signedness of the quantized stored values.
    fn storage_type(&self) -> IntegerTypeRef<'c, 't> {
        unsafe { IntegerTypeRef::from_c_api(mlirQuantizedTypeGetStorageType(self.to_c_api()), self.context()).unwrap() }
    }

    /// Returns the minimum representable storage value for this [`QuantizedType`].
    fn storage_type_minimum(&self) -> i64 {
        unsafe { mlirQuantizedTypeGetStorageTypeMin(self.to_c_api()) }
    }

    /// Returns the maximum representable storage value for this [`QuantizedType`].
    fn storage_type_maximum(&self) -> i64 {
        unsafe { mlirQuantizedTypeGetStorageTypeMax(self.to_c_api()) }
    }

    /// Returns the integral bit width that the underlying storage type of this [`QuantizedType`] can exactly represent.
    /// For integral storage types, this will be just their width.
    fn storage_type_integral_width(&self) -> usize {
        unsafe { mlirQuantizedTypeGetStorageTypeIntegralWidth(self.to_c_api()) as usize }
    }

    /// Returns `true` if `candidate` is a match for this [`QuantizedType`], meaning that the candidate type is either
    /// a primitive type or a container type whose element type equals [`Self::expressed_type`]. Examples of such
    /// compatible expressed types:
    ///
    /// ```text
    /// !quant.uniform<i8:f32, 1.0> =~ f32
    /// !quant.uniform<i8:f32, 1.0> =~ tensor<4xf32>
    /// ```
    fn is_compatible_expressed_type<T: Type<'c, 't>>(&self, candidate: T) -> bool {
        unsafe { mlirQuantizedTypeIsCompatibleExpressedType(self.to_c_api(), candidate.to_c_api()) }
    }

    /// Returns the element type of this [`QuantizedType`] (or [`None`] if it is not a [`QuantizedType`]). If this
    /// type is a primitive type, then it is returned as-is. If it is a container type, then the element type of that
    /// container type will be returned. Examples:
    ///
    /// ```text
    /// !quant.uniform<i8:f32, 1.0> -> !quant.uniform<i8:f32, 1.0>
    /// tensor<4x!quant.uniform<i8:f32, 1.0> -> quant.uniform<i8:f32, 1.0>
    /// ```
    fn quantized_element_type(&self) -> Option<TypeRef<'c, 't>> {
        unsafe { TypeRef::from_c_api(mlirQuantizedTypeGetQuantizedElementType(self.to_c_api()), self.context()) }
    }

    /// Casts the provided `candidate` based on [`Self::storage_type`] to an equivalent [`QuantizedType`], returning
    /// [`None`] if the cast is invalid. This is effectively the inverse of [`Self::cast_to_storage_type`]. Examples
    /// of `candidate` types mapped to their corresponding returned types assuming that the current [`QuantizedType`]
    /// is `!quant.uniform<i8:f32, 1.0>`:
    ///
    /// ```text
    /// i8 -> !quant.uniform<i8:f32, 1.0>
    /// tensor<4xi8> -> tensor<4x!quant.uniform<i8:f32, 1.0}>>
    /// vector<4xi8> -> vector<4x!quant.uniform<i8:f32, 1.0>>
    /// ```
    fn cast_from_storage_type<T: Type<'c, 't>>(&self, candidate: T) -> Option<TypeRef<'c, 't>> {
        unsafe {
            TypeRef::from_c_api(
                mlirQuantizedTypeCastFromStorageType(self.to_c_api(), candidate.to_c_api()),
                self.context(),
            )
        }
    }

    /// Casts this [`QuantizedType`] to a corresponding type based on [`Self::storage_type`].
    /// This is effectively the inverse of [`Self::cast_from_storage_type`].
    fn cast_to_storage_type(&self) -> Option<TypeRef<'c, 't>> {
        unsafe { TypeRef::from_c_api(mlirQuantizedTypeCastToStorageType(self.to_c_api()), self.context()) }
    }

    /// Casts the provided `candidate` based on [`Self::expressed_type`] to an equivalent [`QuantizedType`], returning
    /// [`None`] if the cast is invalid. This is effectively the inverse of [`Self::cast_to_expressed_type`]. Examples
    /// of `candidate` types mapped to their corresponding returned types assuming that the current [`QuantizedType`]
    /// is `!quant.uniform<i8:f32, 1.0>`:
    ///
    /// ```text
    /// f32 -> !quant.uniform<i8:f32, 1.0>
    /// tensor<4xf32> -> tensor<4x!quant.uniform<i8:f32, 1.0>>
    /// vector<4xf32> -> vector<4x!quant.uniform<i8:f32, 1.0>>
    /// ```
    fn cast_from_expressed_type<T: Type<'c, 't>>(&self, candidate: T) -> Option<TypeRef<'c, 't>> {
        unsafe {
            TypeRef::from_c_api(
                mlirQuantizedTypeCastFromExpressedType(self.to_c_api(), candidate.to_c_api()),
                self.context(),
            )
        }
    }

    /// Casts this [`QuantizedType`] to a corresponding type based on [`Self::expressed_type`].
    /// This is effectively the inverse of [`Self::cast_from_expressed_type`].
    fn cast_to_expressed_type(&self) -> Option<TypeRef<'c, 't>> {
        unsafe { TypeRef::from_c_api(mlirQuantizedTypeCastToExpressedType(self.to_c_api()), self.context()) }
    }

    /// Casts the provided `candidate` based on [`Self::expressed_type`] to an equivalent type based on
    /// [`Self::storage_type`], returning [`None`] if the case is invalid. This is equivalent to the composition of
    /// [`Self::cast_to_storage_type`] and [`Self::cast_from_expressed_type`], but with additional validity checks.
    /// Examples of `candidate` types mapped to their corresponding returned types assuming that the current
    /// [`QuantizedType`] is `!quant.uniform<i8:f32, 1.0>`:
    ///
    /// ```text
    /// tensor<4xf32> -> tensor<4xi8>
    /// ```
    fn cast_expressed_to_storage_type<T: Type<'c, 't>>(&self, candidate: T) -> Option<TypeRef<'c, 't>> {
        unsafe {
            TypeRef::from_c_api(
                mlirQuantizedTypeCastExpressedToStorageType(self.to_c_api(), candidate.to_c_api()),
                self.context(),
            )
        }
    }
}

/// Reference to an MLIR `quant::QuantizedType`, the erased supertype of all [`QuantizedType`]s.
///
/// At this level, the type only captures the shared quantization contract:
/// - a storage domain (i.e., [`QuantizedType::storage_type`]) where values are physically represented, and
/// - an expressed domain (i.e., [`QuantizedType::expressed_type`]) where values are interpreted.
///
/// Concrete quantization schemes are represented by specialized subtypes such as [`AnyQuantizedTypeRef`],
/// [`UniformQuantizedTypeRef`], [`UniformQuantizedPerAxisTypeRef`], [`UniformQuantizedSubChannelTypeRef`],
/// and [`CalibratedQuantizedTypeRef`].
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Quantization/#affine-values) and
/// [`quant` dialect documentation](https://mlir.llvm.org/docs/Dialects/QuantDialect/) for more information.
#[derive(Copy, Clone)]
pub struct QuantizedTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> QuantizedType<'c, 't> for QuantizedTypeRef<'c, 't> {}

mlir_subtype_trait_impls!(QuantizedTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = QuantizedType);

/// Reference to an MLIR `quant::AnyQuantizedType`. This is the most permissive concrete [`QuantizedType`] in the
/// `quant` dialect. It preserves the common [`QuantizedType`] metadata (i.e., flags, storage constraints, and expressed
/// type compatibility), but it does not encode an explicit quantization mapping (i.e., no fixed scale or zero-point
/// parameters). Use this type as a placeholder when a value is known to be quantized but its final concrete
/// quantization scheme has not been committed yet.
///
/// # Examples
///
/// ```text
/// !quant.any<i8:f32>
/// !quant.any<i8<-127:127>:f32>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Quantization/) for more information.
#[derive(Copy, Clone)]
pub struct AnyQuantizedTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> AnyQuantizedTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`AnyQuantizedTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirAnyQuantizedTypeGetTypeID()).unwrap() }
    }
}

impl<'c, 't> QuantizedType<'c, 't> for AnyQuantizedTypeRef<'c, 't> {}

mlir_subtype_trait_impls!(AnyQuantizedTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = AnyQuantizedType);

impl<'t> Context<'t> {
    /// Constructs a new [`AnyQuantizedTypeRef`]. Refer to the documentation
    /// of that type for information on the arguments of this function.
    pub fn any_quantized_type<'c, StorageType: Type<'c, 't>, ExpressedType: Type<'c, 't>>(
        &'c self,
        flags: u32,
        storage_type: StorageType,
        expressed_type: ExpressedType,
        storage_type_min: i64,
        storage_type_max: i64,
    ) -> AnyQuantizedTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::quant());
        unsafe {
            AnyQuantizedTypeRef::from_c_api(
                mlirAnyQuantizedTypeGet(
                    flags,
                    storage_type.to_c_api(),
                    expressed_type.to_c_api(),
                    storage_type_min,
                    storage_type_max,
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Reference to an MLIR `quant::UniformQuantizedType` (i.e., using per-layer/per-tensor quantization). This type
/// applies one affine quantization mapping to all values: `expressed = (stored - zero_point) * scale`, referencing
/// [`UniformQuantizedTypeRef::scale`] and [`UniformQuantizedTypeRef::zero_point`]. Unlike
/// [`UniformQuantizedPerAxisTypeRef`] and [`UniformQuantizedSubChannelTypeRef`], the quantization
/// parameters are not index-dependent.
///
/// # Examples
///
/// ```text
/// !quant.uniform<i8:f32, 5.000000e-01:5>
/// !quant.uniform<i8<-127:127>:f32, 1.000000e+00:0>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/QuantDialect/#per-layer-quantization)
/// for more information.
#[derive(Copy, Clone)]
pub struct UniformQuantizedTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> UniformQuantizedTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`UniformQuantizedTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirUniformQuantizedTypeGetTypeID()).unwrap() }
    }

    /// Returns `true` if this type represents fixed-point quantization. In MLIR, this means that the storage and
    /// expressed types are both integer types with the same signedness, and the storage width is strictly larger
    /// than the expressed width.
    pub fn is_fixed_point(&self) -> bool {
        unsafe { mlirUniformQuantizedTypeIsFixedPoint(self.handle) }
    }

    /// Returns the single scale used for all quantized values of this type.
    pub fn scale(&self) -> f64 {
        unsafe { mlirUniformQuantizedTypeGetScale(self.handle) }
    }

    /// Returns the single zero-point used for all quantized values of this type.
    pub fn zero_point(&self) -> i64 {
        unsafe { mlirUniformQuantizedTypeGetZeroPoint(self.handle) }
    }
}

impl<'c, 't> QuantizedType<'c, 't> for UniformQuantizedTypeRef<'c, 't> {}

mlir_subtype_trait_impls!(
    UniformQuantizedTypeRef<'c, 't> as Type,
    mlir_type = Type,
    mlir_subtype = UniformQuantizedType,
);

impl<'t> Context<'t> {
    /// Constructs a new [`UniformQuantizedTypeRef`]. Refer to the documentation
    /// of that type for information on the arguments of this function.
    #[allow(clippy::too_many_arguments)]
    pub fn uniform_quantized_type<'c, StorageType: Type<'c, 't>, ExpressedType: Type<'c, 't>>(
        &'c self,
        flags: u32,
        storage_type: StorageType,
        expressed_type: ExpressedType,
        scale: f64,
        zero_point: i64,
        storage_type_min: i64,
        storage_type_max: i64,
    ) -> UniformQuantizedTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::quant());
        unsafe {
            UniformQuantizedTypeRef::from_c_api(
                mlirUniformQuantizedTypeGet(
                    flags,
                    storage_type.to_c_api(),
                    expressed_type.to_c_api(),
                    scale,
                    zero_point,
                    storage_type_min,
                    storage_type_max,
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Reference to an MLIR `quant::UniformQuantizedPerAxisType` (i.e., using per-channel/per-axis quantization). This
/// type extends [`UniformQuantizedTypeRef`] by using one `(scale, zero-point)` pair per index along a chosen quantized
/// dimension. If `i` indexes that dimension, the corresponding mapping is:
/// `expressed = (stored - zero_point[i]) * scale[i]`. Use this when quantization parameters
/// vary by channel/axis but are constant within each slice.
///
/// # Examples
///
/// ```text
/// tensor<2x3x4x!quant.uniform<i8:f32:1, {3.0, 4.0, 5.0}>>
/// tensor<?x?x!quant.uniform<u16:f32:0, {2.0:10, 3.0:20}>>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/QuantDialect/#per-channel-quantization)
/// for more information.
#[derive(Copy, Clone)]
pub struct UniformQuantizedPerAxisTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> UniformQuantizedPerAxisTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`UniformQuantizedPerAxisTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirUniformQuantizedPerAxisTypeGetTypeID()).unwrap() }
    }

    /// Returns `true` if this type represents fixed-point quantization. In MLIR, this means that the storage and
    /// expressed types are both integer types with the same signedness, and the storage width is strictly larger
    /// than the expressed width.
    pub fn is_fixed_point(&self) -> bool {
        unsafe { mlirUniformQuantizedPerAxisTypeIsFixedPoint(self.handle) }
    }

    /// Returns the number of quantized dimensions of this [`UniformQuantizedPerAxisTypeRef`]
    /// (i.e., the number of per-axis parameter entries).
    pub fn quantized_dimension_count(&self) -> usize {
        unsafe { mlirUniformQuantizedPerAxisTypeGetNumDims(self.handle).cast_unsigned() }
    }

    /// Returns the `index`-th quantized dimension index that is to be paired with the `index`-th scale and zero point
    /// parameters of this [`UniformQuantizedPerAxisTypeRef`]. This function will return [`None`] if `index` is out of
    /// bounds. Note that, for per-axis quantization in MLIR, the quantized dimension is shared across all entries and
    /// therefore the same value is returned for all valid indices.
    pub fn quantized_dimension(&self, index: usize) -> Option<usize> {
        if index >= self.quantized_dimension_count() {
            return None;
        }
        Some(unsafe { mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(self.handle).cast_unsigned() as usize })
    }

    /// Returns all quantized dimension indices of this [`UniformQuantizedPerAxisTypeRef`]
    /// (i.e., one for each per-axis parameter entry).
    pub fn quantized_dimensions(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.quantized_dimension_count()).map(|index| self.quantized_dimension(index).unwrap())
    }

    /// Returns the `index`-th scale parameter of this [`UniformQuantizedPerAxisTypeRef`], and [`None`]
    /// if the provided `index` is out of bounds.
    pub fn scale(&self, index: usize) -> Option<f64> {
        if index >= self.quantized_dimension_count() {
            return None;
        }
        Some(unsafe { mlirUniformQuantizedPerAxisTypeGetScale(self.handle, index.cast_signed()) })
    }

    /// Returns all the per-axis scale parameters of this [`UniformQuantizedPerAxisTypeRef`],
    /// ordered by index along [`Self::quantized_dimensions`].
    pub fn scales(&self) -> impl Iterator<Item = f64> + '_ {
        (0..self.quantized_dimension_count()).map(|index| self.scale(index).unwrap())
    }

    /// Returns the `index`-th zero-point parameter of this [`UniformQuantizedPerAxisTypeRef`], and [`None`]
    /// if the provided `index` is out of bounds.
    pub fn zero_point(&self, index: usize) -> Option<i64> {
        if index >= self.quantized_dimension_count() {
            return None;
        }
        Some(unsafe { mlirUniformQuantizedPerAxisTypeGetZeroPoint(self.handle, index.cast_signed()) })
    }

    /// Returns all the per-axis zero-point parameters of this [`UniformQuantizedPerAxisTypeRef`],
    /// ordered by index along [`Self::quantized_dimensions`].
    pub fn zero_points(&self) -> impl Iterator<Item = i64> + '_ {
        (0..self.quantized_dimension_count()).map(|index| self.zero_point(index).unwrap())
    }
}

impl<'c, 't> QuantizedType<'c, 't> for UniformQuantizedPerAxisTypeRef<'c, 't> {}

mlir_subtype_trait_impls!(
    UniformQuantizedPerAxisTypeRef<'c, 't> as Type,
    mlir_type = Type,
    mlir_subtype = UniformQuantizedPerAxisType,
);

impl<'t> Context<'t> {
    /// Constructs a new [`UniformQuantizedPerAxisTypeRef`]. Refer to the documentation
    /// of that type for information on the arguments of this function.
    #[allow(clippy::too_many_arguments)]
    pub fn uniform_quantized_per_axis_type<'c, StorageType: Type<'c, 't>, ExpressedType: Type<'c, 't>>(
        &'c self,
        flags: u32,
        storage_type: StorageType,
        expressed_type: ExpressedType,
        scales: &[f64],
        zero_points: &[i64],
        quantized_dimension: i32,
        storage_type_min: i64,
        storage_type_max: i64,
    ) -> UniformQuantizedPerAxisTypeRef<'c, 't> {
        if scales.len() != zero_points.len() {
            panic!("`scales` and `zero_points` have the same length");
        }
        self.load_dialect(DialectHandle::quant());
        unsafe {
            UniformQuantizedPerAxisTypeRef::from_c_api(
                mlirUniformQuantizedPerAxisTypeGet(
                    flags,
                    storage_type.to_c_api(),
                    expressed_type.to_c_api(),
                    scales.len().cast_signed(),
                    scales.as_ptr() as *mut _,
                    zero_points.as_ptr() as *mut _,
                    quantized_dimension,
                    storage_type_min,
                    storage_type_max,
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Reference to an MLIR `quant::UniformQuantizedSubChannelType` (i.e., using block-wise quantization). This is
/// a finer-grained generalization of [`UniformQuantizedPerAxisTypeRef`]. Instead of one parameter pair per full
/// channel/axis slice, parameters are selected per block across one or more quantized dimensions (i.e.,
/// `quantized_dimensions` with corresponding `block_sizes`). The `scales` and `zero_points` values are
/// stored as dense elements attributes and are indexed by the computed block coordinates. Upstream MLIR
/// currently supports this form for ranked tensor values.
///
/// # Examples
///
/// ```text
/// tensor<3x4x!quant.uniform<i8:f32:{0:1, 1:2}, {{2.0:10, 3.0:20}, {4.0:30, 5.0:40}, {6.0:50, 7.0:60}}>>
/// tensor<?x?x!quant.uniform<u16:f32:{0:1, 1:2}, {{2.0:10, 3.0:20}, {4.0:30, 5.0:40}}>>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/QuantDialect/#sub-channel-quantization)
/// for more information.
#[derive(Copy, Clone)]
pub struct UniformQuantizedSubChannelTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> UniformQuantizedSubChannelTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`UniformQuantizedSubChannelTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirUniformQuantizedSubChannelTypeGetTypeID()).unwrap() }
    }

    /// Returns the number of blocks of this [`UniformQuantizedSubChannelTypeRef`]
    /// (i.e., the number of `(quantized_dimension, block_size)` entries).
    pub fn block_count(&self) -> usize {
        unsafe { mlirUniformQuantizedSubChannelTypeGetNumBlockSizes(self.handle).cast_unsigned() }
    }

    /// Returns the `index`-th quantized dimension index that is to be paired with the `index`-th quantization block
    /// parameters of this [`UniformQuantizedSubChannelTypeRef`]. This function will return [`None`] if `index` is out
    /// of bounds.
    pub fn quantized_dimension(&self, index: usize) -> Option<usize> {
        if index >= self.block_count() {
            return None;
        }
        Some(unsafe {
            mlirUniformQuantizedSubChannelTypeGetQuantizedDimension(self.handle, index.cast_signed()).cast_unsigned()
                as usize
        })
    }

    /// Returns all quantized dimension indices of this [`UniformQuantizedPerAxisTypeRef`]
    /// (i.e., one for each quantization block).
    pub fn quantized_dimensions(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.block_count()).map(|index| self.quantized_dimension(index).unwrap())
    }

    /// Returns the `index`-th block size parameter of this [`UniformQuantizedPerAxisTypeRef`], and [`None`]
    /// if the provided `index` is out of bounds.
    pub fn block_size(&self, index: usize) -> Option<i64> {
        if index >= self.block_count() {
            return None;
        }
        Some(unsafe { mlirUniformQuantizedSubChannelTypeGetBlockSize(self.handle, index.cast_signed()) })
    }

    /// Returns all the block size parameters of this [`UniformQuantizedSubChannelTypeRef`],
    /// ordered by index along [`Self::quantized_dimensions`].
    pub fn block_sizes(&self) -> impl Iterator<Item = i64> + '_ {
        (0..self.block_count()).map(|index| self.block_size(index).unwrap())
    }

    /// Returns the [`AttributeRef`] that stores sub-channel scales for this [`UniformQuantizedSubChannelTypeRef`].
    pub fn scales(&self) -> AttributeRef<'c, 't> {
        unsafe {
            AttributeRef::from_c_api(mlirUniformQuantizedSubChannelTypeGetScales(self.handle), self.context).unwrap()
        }
    }

    /// Returns the [`AttributeRef`] that stores sub-channel zero-points for this [`UniformQuantizedSubChannelTypeRef`].
    pub fn zero_points(&self) -> AttributeRef<'c, 't> {
        unsafe {
            AttributeRef::from_c_api(mlirUniformQuantizedSubChannelTypeGetZeroPoints(self.handle), self.context)
                .unwrap()
        }
    }
}

impl<'c, 't> QuantizedType<'c, 't> for UniformQuantizedSubChannelTypeRef<'c, 't> {}

mlir_subtype_trait_impls!(
    UniformQuantizedSubChannelTypeRef<'c, 't> as Type,
    mlir_type = Type,
    mlir_subtype = UniformQuantizedSubChannelType,
);

impl<'t> Context<'t> {
    /// Constructs a new [`UniformQuantizedSubChannelTypeRef`]. Refer to the documentation
    /// of that type for information on the arguments of this function.
    #[allow(clippy::too_many_arguments)]
    pub fn uniform_quantized_sub_channel_type<
        'c,
        StorageType: Type<'c, 't>,
        ExpressedType: Type<'c, 't>,
        Scales: Attribute<'c, 't>,
        ZeroPoints: Attribute<'c, 't>,
    >(
        &'c self,
        flags: u32,
        storage_type: StorageType,
        expressed_type: ExpressedType,
        scales: Scales,
        zero_points: ZeroPoints,
        quantized_dimensions: &[i32],
        block_sizes: &[i64],
        storage_type_min: i64,
        storage_type_max: i64,
    ) -> UniformQuantizedSubChannelTypeRef<'c, 't> {
        if quantized_dimensions.len() != block_sizes.len() {
            panic!("`quantized_dimensions` and `block_sizes` must have the same length");
        }
        self.load_dialect(DialectHandle::quant());
        unsafe {
            UniformQuantizedSubChannelTypeRef::from_c_api(
                mlirUniformQuantizedSubChannelTypeGet(
                    flags,
                    storage_type.to_c_api(),
                    expressed_type.to_c_api(),
                    scales.to_c_api(),
                    zero_points.to_c_api(),
                    quantized_dimensions.len().cast_signed(),
                    quantized_dimensions.as_ptr() as *mut _,
                    block_sizes.as_ptr() as *mut _,
                    storage_type_min,
                    storage_type_max,
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Reference to an MLIR `quant::CalibratedQuantizedType`. This type carries only calibration statistics for an
/// expressed floating-point type, represented by the observed real-value interval `[minimum, maximum]`. It does
/// not commit to a concrete storage mapping (e.g., scale or zero-point parameters), and is typically used as an
/// intermediate annotation before lowering to a concrete uniform quantized type. MLIR verifies that
/// [`CalibratedQuantizedTypeRef::minimum`] is less than or equal to [`CalibratedQuantizedTypeRef::maximum`]
/// and that the expressed type is floating-point.
///
/// # Examples
///
/// ```text
/// !quant.calibrated<f32<-0.998:1.2321>>
/// tensor<4x!quant.calibrated<f32<-1.0:1.0>>>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Quantization/) for more information.
#[derive(Copy, Clone)]
pub struct CalibratedQuantizedTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> CalibratedQuantizedTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`CalibratedQuantizedTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirCalibratedQuantizedTypeGetTypeID()).unwrap() }
    }

    /// Returns the minimum calibrated value of this [`CalibratedQuantizedTypeRef`].
    pub fn minimum(&self) -> f64 {
        unsafe { mlirCalibratedQuantizedTypeGetMin(self.handle) }
    }

    /// Returns the maximum calibrated value of this [`CalibratedQuantizedTypeRef`].
    pub fn maximum(&self) -> f64 {
        unsafe { mlirCalibratedQuantizedTypeGetMax(self.handle) }
    }
}

impl<'c, 't> QuantizedType<'c, 't> for CalibratedQuantizedTypeRef<'c, 't> {}

mlir_subtype_trait_impls!(
    CalibratedQuantizedTypeRef<'c, 't> as Type,
    mlir_type = Type,
    mlir_subtype = CalibratedQuantizedType,
);

impl<'t> Context<'t> {
    /// Constructs a new [`CalibratedQuantizedTypeRef`]. Refer to the documentation
    /// of that type for information on the arguments of this function.
    pub fn calibrated_quantized_type<'c, ExpressedType: Type<'c, 't>>(
        &'c self,
        expressed_type: ExpressedType,
        minimum: f64,
        maximum: f64,
    ) -> CalibratedQuantizedTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::quant());
        unsafe {
            CalibratedQuantizedTypeRef::from_c_api(
                mlirCalibratedQuantizedTypeGet(expressed_type.to_c_api(), minimum, maximum),
                self,
            )
            .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Size;
    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_any_quantized_type() {
        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        assert_eq!(QuantizedTypeRef::default_minimum_for_integer(true, 8), -128);
        assert_eq!(QuantizedTypeRef::default_maximum_for_integer(true, 8), 127);
        let r#type = context.any_quantized_type(flags, storage_type, expressed_type, -128, 127);
        assert_eq!(r#type.dialect(), context.load_dialect(DialectHandle::quant()).unwrap());
        assert_eq!(r#type.type_id(), AnyQuantizedTypeRef::type_id());
        assert_eq!(r#type.flags(), flags);
        assert_eq!(r#type.storage_type(), storage_type);
        assert_eq!(r#type.expressed_type(), expressed_type);
        assert_eq!(r#type.storage_type_minimum(), -128);
        assert_eq!(r#type.storage_type_maximum(), 127);
    }

    #[test]
    fn test_any_quantized_type_equality() {
        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let type_1 = context.any_quantized_type(flags, storage_type, expressed_type, -128, 127);
        let type_2 = context.any_quantized_type(flags, storage_type, expressed_type, -128, 127);
        assert_eq!(type_1, type_2);

        let type_2 = context.any_quantized_type(flags, storage_type, expressed_type, -127, 127);
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let type_2 = context.any_quantized_type(flags, storage_type, expressed_type, -128, 127);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_any_quantized_type_display_and_debug() {
        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let r#type = context.any_quantized_type(flags, storage_type, expressed_type, -128, 127);
        test_type_display_and_debug(r#type, "!quant.any<i8:f32>");
    }

    #[test]
    fn test_any_quantized_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::quant());
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let r#type = context.any_quantized_type(flags, storage_type, expressed_type, -127, 127);
        assert_eq!(context.parse_type("!quant.any<i8<-127:127>:f32>").unwrap(), r#type);
    }

    #[test]
    fn test_any_quantized_type_casting() {
        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let r#type = context.any_quantized_type(flags, storage_type, expressed_type, -128, 127);
        test_type_casting(r#type);
    }

    #[test]
    fn test_uniform_quantized_type() {
        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let r#type = context.uniform_quantized_type(flags, storage_type, expressed_type, 0.25, 3, -128, 127);
        assert_eq!(r#type.dialect(), context.load_dialect(DialectHandle::quant()).unwrap());
        assert_eq!(r#type.type_id(), UniformQuantizedTypeRef::type_id());
        assert!(!r#type.is_fixed_point());
        assert_eq!(r#type.scale(), 0.25);
        assert_eq!(r#type.zero_point(), 3);
    }

    #[test]
    fn test_uniform_quantized_type_equality() {
        let context = Context::new();

        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let type_1 = context.uniform_quantized_type(flags, storage_type, expressed_type, 0.25, 3, -128, 127);
        let type_2 = context.uniform_quantized_type(flags, storage_type, expressed_type, 0.25, 3, -128, 127);
        assert_eq!(type_1, type_2);

        let type_2 = context.uniform_quantized_type(flags, storage_type, expressed_type, 0.5, 3, -128, 127);
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let type_2 = context.uniform_quantized_type(flags, storage_type, expressed_type, 0.25, 3, -128, 127);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_uniform_quantized_type_display_and_debug() {
        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let r#type = context.uniform_quantized_type(flags, storage_type, expressed_type, 0.25, 3, -127, 127);
        test_type_display_and_debug(r#type, "!quant.uniform<i8<-127:127>:f32, 2.500000e-01:3>");
    }

    #[test]
    fn test_uniform_quantized_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::quant());
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let r#type = context.uniform_quantized_type(flags, storage_type, expressed_type, 0.25, 3, -128, 127);
        assert_eq!(context.parse_type("!quant.uniform<i8:f32, 2.500000e-01:3>").unwrap(), r#type);
    }

    #[test]
    fn test_uniform_quantized_type_casting() {
        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let r#type = context.uniform_quantized_type(flags, storage_type, expressed_type, 0.25, 3, -128, 127);
        test_type_casting(r#type);
    }

    #[test]
    fn test_uniform_quantized_per_axis_type() {
        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let r#type = context.uniform_quantized_per_axis_type(
            flags,
            storage_type,
            expressed_type,
            &[0.25, 0.5],
            &[3, 5],
            1,
            -128,
            127,
        );
        assert_eq!(r#type.dialect(), context.load_dialect(DialectHandle::quant()).unwrap());
        assert_eq!(r#type.type_id(), UniformQuantizedPerAxisTypeRef::type_id());
        assert!(r#type.is_fixed_point());
        assert_eq!(r#type.quantized_dimension_count(), 2);
        assert_eq!(r#type.quantized_dimension(0), Some(1));
        assert_eq!(r#type.quantized_dimension(1), Some(1));
        assert_eq!(r#type.quantized_dimension(2), None);
        assert_eq!(r#type.quantized_dimensions().collect::<Vec<_>>(), vec![1, 1]);
        assert_eq!(r#type.scale(0), Some(0.25));
        assert_eq!(r#type.scale(1), Some(0.5));
        assert_eq!(r#type.scale(2), None);
        assert_eq!(r#type.scales().collect::<Vec<_>>(), vec![0.25, 0.5]);
        assert_eq!(r#type.zero_point(0), Some(3));
        assert_eq!(r#type.zero_point(1), Some(5));
        assert_eq!(r#type.zero_point(2), None);
        assert_eq!(r#type.zero_points().collect::<Vec<_>>(), vec![3, 5]);
    }

    #[test]
    fn test_uniform_quantized_per_axis_type_equality() {
        let context = Context::new();

        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let type_1 = context.uniform_quantized_per_axis_type(
            flags,
            storage_type,
            expressed_type,
            &[0.25, 0.5],
            &[3, 5],
            1,
            -128,
            127,
        );
        let type_2 = context.uniform_quantized_per_axis_type(
            flags,
            storage_type,
            expressed_type,
            &[0.25, 0.5],
            &[3, 5],
            1,
            -128,
            127,
        );
        assert_eq!(type_1, type_2);

        let type_2 = context.uniform_quantized_per_axis_type(
            flags,
            storage_type,
            expressed_type,
            &[0.25, 0.5],
            &[3, 5],
            0,
            -128,
            127,
        );
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let type_2 = context.uniform_quantized_per_axis_type(
            flags,
            storage_type,
            expressed_type,
            &[0.25, 0.5],
            &[3, 5],
            1,
            -128,
            127,
        );
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_uniform_quantized_per_axis_type_display_and_debug() {
        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let r#type = context.uniform_quantized_per_axis_type(
            flags,
            storage_type,
            expressed_type,
            &[0.25, 0.5],
            &[3, 5],
            1,
            -128,
            127,
        );
        test_type_display_and_debug(r#type, "!quant.uniform<i8:f32:1, {2.500000e-01:3,5.000000e-01:5}>");
    }

    #[test]
    fn test_uniform_quantized_per_axis_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::quant());
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let r#type = context.uniform_quantized_per_axis_type(
            flags,
            storage_type,
            expressed_type,
            &[0.25, 0.5],
            &[3, 5],
            1,
            -128,
            127,
        );
        assert_eq!(context.parse_type("!quant.uniform<i8:f32:1, {2.500000e-01:3,5.000000e-01:5}>").unwrap(), r#type);
    }

    #[test]
    fn test_uniform_quantized_per_axis_type_casting() {
        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let r#type = context.uniform_quantized_per_axis_type(
            flags,
            storage_type,
            expressed_type,
            &[0.25, 0.5],
            &[3, 5],
            1,
            -128,
            127,
        );
        test_type_casting(r#type);
    }

    #[test]
    fn test_uniform_quantized_sub_channel_type() {
        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let location = context.unknown_location();
        let scale_type = context.tensor_type(context.float32_type(), &[Size::Static(2)], None, location).unwrap();
        let zero_point_type =
            context.tensor_type(context.signless_integer_type(64), &[Size::Static(2)], None, location).unwrap();
        let scales = context.dense_f32_elements_attribute(scale_type, &[0.25, 0.5]).unwrap();
        let zero_points = context.dense_i64_elements_attribute(zero_point_type, &[0, 2]).unwrap();
        let scales_display = scales.to_string();
        let zero_points_display = zero_points.to_string();
        let r#type = context.uniform_quantized_sub_channel_type(
            flags,
            storage_type,
            expressed_type,
            scales,
            zero_points,
            &[0, 1],
            &[16, 8],
            -128,
            127,
        );
        assert_eq!(r#type.dialect(), context.load_dialect(DialectHandle::quant()).unwrap());
        assert_eq!(r#type.type_id(), UniformQuantizedSubChannelTypeRef::type_id());
        assert_eq!(r#type.block_count(), 2);
        assert_eq!(r#type.quantized_dimension(0), Some(0));
        assert_eq!(r#type.quantized_dimension(1), Some(1));
        assert_eq!(r#type.quantized_dimension(2), None);
        assert_eq!(r#type.quantized_dimensions().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(r#type.block_size(0), Some(16));
        assert_eq!(r#type.block_size(1), Some(8));
        assert_eq!(r#type.block_size(2), None);
        assert_eq!(r#type.block_sizes().collect::<Vec<_>>(), vec![16, 8]);
        assert_eq!(r#type.scales().to_string(), scales_display);
        assert_eq!(r#type.zero_points().to_string(), zero_points_display);
    }

    #[test]
    fn test_uniform_quantized_sub_channel_type_equality() {
        let context = Context::new();

        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let location = context.unknown_location();
        let scale_type = context.tensor_type(context.float32_type(), &[Size::Static(2)], None, location).unwrap();
        let zero_point_type =
            context.tensor_type(context.signless_integer_type(64), &[Size::Static(2)], None, location).unwrap();
        let scales = context.dense_f32_elements_attribute(scale_type, &[0.25, 0.5]).unwrap();
        let zero_points = context.dense_i64_elements_attribute(zero_point_type, &[0, 2]).unwrap();
        let type_1 = context.uniform_quantized_sub_channel_type(
            flags,
            storage_type,
            expressed_type,
            scales,
            zero_points,
            &[0, 1],
            &[16, 8],
            -128,
            127,
        );
        let scale_type = context.tensor_type(context.float32_type(), &[Size::Static(2)], None, location).unwrap();
        let zero_point_type =
            context.tensor_type(context.signless_integer_type(64), &[Size::Static(2)], None, location).unwrap();
        let scales = context.dense_f32_elements_attribute(scale_type, &[0.25, 0.5]).unwrap();
        let zero_points = context.dense_i64_elements_attribute(zero_point_type, &[0, 2]).unwrap();
        let type_2 = context.uniform_quantized_sub_channel_type(
            flags,
            storage_type,
            expressed_type,
            scales,
            zero_points,
            &[0, 1],
            &[16, 8],
            -128,
            127,
        );
        assert_eq!(type_1, type_2);

        let scale_type = context.tensor_type(context.float32_type(), &[Size::Static(2)], None, location).unwrap();
        let zero_point_type =
            context.tensor_type(context.signless_integer_type(64), &[Size::Static(2)], None, location).unwrap();
        let scales = context.dense_f32_elements_attribute(scale_type, &[0.25, 0.5]).unwrap();
        let zero_points = context.dense_i64_elements_attribute(zero_point_type, &[0, 2]).unwrap();
        let type_2 = context.uniform_quantized_sub_channel_type(
            flags,
            context.signless_integer_type(8),
            context.float32_type(),
            scales,
            zero_points,
            &[0, 1],
            &[8, 8],
            -128,
            127,
        );
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let location = context.unknown_location();
        let scale_type = context.tensor_type(context.float32_type(), &[Size::Static(2)], None, location).unwrap();
        let zero_point_type =
            context.tensor_type(context.signless_integer_type(64), &[Size::Static(2)], None, location).unwrap();
        let scales = context.dense_f32_elements_attribute(scale_type, &[0.25, 0.5]).unwrap();
        let zero_points = context.dense_i64_elements_attribute(zero_point_type, &[0, 2]).unwrap();
        let type_2 = context.uniform_quantized_sub_channel_type(
            flags,
            storage_type,
            expressed_type,
            scales,
            zero_points,
            &[0, 1],
            &[16, 8],
            -128,
            127,
        );
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_uniform_quantized_sub_channel_type_display_and_debug() {
        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let location = context.unknown_location();
        let scale_type = context.tensor_type(context.float32_type(), &[Size::Static(2)], None, location).unwrap();
        let zero_point_type =
            context.tensor_type(context.signless_integer_type(64), &[Size::Static(2)], None, location).unwrap();
        let scales = context.dense_f32_elements_attribute(scale_type, &[0.25, 0.5]).unwrap();
        let zero_points = context.dense_i64_elements_attribute(zero_point_type, &[0, 2]).unwrap();
        let r#type = context.uniform_quantized_sub_channel_type(
            flags,
            storage_type,
            expressed_type,
            scales,
            zero_points,
            &[0, 1],
            &[16, 8],
            -128,
            127,
        );
        test_type_display_and_debug(r#type, "!quant.uniform<i8:f32:{0:16, 1:8}, {2.500000e-01, 5.000000e-01:2}>");
    }

    #[test]
    fn test_uniform_quantized_sub_channel_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::quant());
        let parsed = context
            .parse_type("!quant.uniform<i8:f32:{0:1, 1:2}, {{2.0:10, 3.0:20}, {4.0:30, 5.0:40}, {6.0:50, 7.0:60}}>")
            .unwrap();
        assert!(parsed.is::<UniformQuantizedSubChannelTypeRef>());
        let parsed = parsed.cast::<UniformQuantizedSubChannelTypeRef>().unwrap();
        assert_eq!(
            parsed.to_string(),
            "!quant.uniform<\
              i8:f32:{0:1, 1:2}, \
              {\
                {2.000000e+00:10, 3.000000e+00:20}, \
                {4.000000e+00:30, 5.000000e+00:40}, \
                {6.000000e+00:50, 7.000000e+00:60}\
              }\
            >",
        );
    }

    #[test]
    fn test_uniform_quantized_sub_channel_type_casting() {
        let context = Context::new();
        let flags = QuantizedTypeRef::signed_flag();
        let storage_type = context.signless_integer_type(8);
        let expressed_type = context.float32_type();
        let location = context.unknown_location();
        let scale_type = context.tensor_type(context.float32_type(), &[Size::Static(2)], None, location).unwrap();
        let zero_point_type =
            context.tensor_type(context.signless_integer_type(64), &[Size::Static(2)], None, location).unwrap();
        let scales = context.dense_f32_elements_attribute(scale_type, &[0.25, 0.5]).unwrap();
        let zero_points = context.dense_i64_elements_attribute(zero_point_type, &[0, 2]).unwrap();
        let r#type = context.uniform_quantized_sub_channel_type(
            flags,
            storage_type,
            expressed_type,
            scales,
            zero_points,
            &[0, 1],
            &[16, 8],
            -128,
            127,
        );
        test_type_casting(r#type);
    }

    #[test]
    fn test_calibrated_quantized_type() {
        let context = Context::new();
        let expressed_type = context.float32_type();
        let r#type = context.calibrated_quantized_type(expressed_type, -1.0, 1.0);
        assert_eq!(r#type.dialect(), context.load_dialect(DialectHandle::quant()).unwrap());
        assert_eq!(r#type.type_id(), CalibratedQuantizedTypeRef::type_id());
        assert_eq!(r#type.minimum(), -1.0);
        assert_eq!(r#type.maximum(), 1.0);
    }

    #[test]
    fn test_calibrated_quantized_type_equality() {
        let context = Context::new();

        let type_1 = context.calibrated_quantized_type(context.float32_type(), -1.0, 1.0);
        let type_2 = context.calibrated_quantized_type(context.float32_type(), -1.0, 1.0);
        assert_eq!(type_1, type_2);

        let type_2 = context.calibrated_quantized_type(context.float32_type(), -2.0, 1.0);
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.calibrated_quantized_type(context.float32_type(), -1.0, 1.0);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_calibrated_quantized_type_display_and_debug() {
        let context = Context::new();
        let r#type = context.calibrated_quantized_type(context.float32_type(), -1.0, 1.0);
        test_type_display_and_debug(r#type, "!quant.calibrated<f32<-1.000000e+00:1.000000e+00>>");
    }

    #[test]
    fn test_calibrated_quantized_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::quant());
        let r#type = context.calibrated_quantized_type(context.float32_type(), -1.0, 1.0);
        assert_eq!(context.parse_type("!quant.calibrated<f32<-1.000000e+00:1.000000e+00>>").unwrap(), r#type);
    }

    #[test]
    fn test_calibrated_quantized_type_casting() {
        let context = Context::new();
        let r#type = context.calibrated_quantized_type(context.float32_type(), -1.0, 1.0);
        test_type_casting(r#type);
    }
}
