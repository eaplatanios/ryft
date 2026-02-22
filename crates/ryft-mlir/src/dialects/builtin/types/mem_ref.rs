use ryft_xla_sys::bindings::{
    MlirType, mlirMemRefTypeContiguousGetChecked, mlirMemRefTypeGetAffineMap, mlirMemRefTypeGetChecked,
    mlirMemRefTypeGetLayout, mlirMemRefTypeGetMemorySpace, mlirMemRefTypeGetStridesAndOffset, mlirMemRefTypeGetTypeID,
    mlirShapedTypeGetDimSize, mlirShapedTypeGetRank, mlirShapedTypeIsStaticDim, mlirUnrankedMemRefTypeGetChecked,
    mlirUnrankedMemRefTypeGetTypeID, mlirUnrankedMemrefGetMemorySpace,
};

use crate::{
    AffineMap, Attribute, AttributeRef, Context, Location, LogicalResult, Type, TypeId, mlir_subtype_trait_impls,
};

use super::{ShapedType, Size};

/// Built-in MLIR [`Type`] that represents a shaped reference to a region of memory with a known rank (i.e., a fixed
/// number of dimensions). This is in contrast to [`UnrankedMemRefTypeRef`] which is used to represent shaped references
/// to regions of memory with an unknown number of dimensions. [`MemRefTypeRef`]s have the same shape specifiers as
/// [`TensorTypeRef`](crate::TensorTypeRef)s but they also have additional attributes which specify the way in which
/// the underlying data is laid out in memory (e.g., using contiguous, strided, etc., layout).
///
/// [`MemRefTypeRef`] values are references to regions of memory (similar to a buffer pointers but more powerful).
/// The buffer pointed to by a [`MemRefTypeRef`] value can be allocated, aliased and deallocated. A [`MemRefTypeRef`]
/// value can be used to read and write data from/to the memory region that it references.
///
/// The memory space of a [`MemRefTypeRef`] is specified by a target-specific attribute. It might be an integer value,
/// string, dictionary or even custom dialect attribute. The empty memory space, which is represented by a null
/// attribute, is target specific.
///
/// The notionally dynamic value of a [`MemRefTypeRef`] value includes the address of the buffer allocated,
/// as well as the symbols referred to by the shape, layout map, and index maps.
///
/// # Layout
///
/// A [`MemRefTypeRef`] may optionally have a layout that indicates how indices are transformed from the
/// multidimensional form into a linear address. The layout must avoid internal aliasing (i.e., two distinct tuples
/// of in-bounds indices must be pointing to different elements in memory). The layout is an attribute that implements.
/// The built-in dialect offers two kinds of layouts: strided and [`AffineMap`]s, each of which is available as an
/// attribute. Other attributes may be used to represent the layout as long as they can be converted to a
/// [semi-affine map](https://mlir.llvm.org/docs/Dialects/Affine/#semi-affine-maps) and implement the required
/// interface. Users of [`MemRefTypeRef`]s are expected to fall back to the affine representation when handling
/// unknown layouts. Multidimensional affine forms are interpreted in a row-major fashion.
///
/// In absence of an explicit layout, a [`MemRefTypeRef`] is considered to have a multidimensional identity affine
/// map layout. Identity layout maps do not contribute to the [`MemRefTypeRef`] identification and are discarded on
/// construction. That is, a type with an explicit identity map like `memref<?x?xf32, (i,j)->(i,j)>` is strictly the
/// same as this one without a layout `memref<?x?xf32>`.
///
/// The core syntax and representation of a layout specification is a
/// [semi-affine map](https://mlir.llvm.org/docs/Dialects/Affine/#semi-affine-maps). Additionally, syntactic sugar is
/// supported to make certain layout specifications more intuitive to read. For the moment, [`MemRefTypeRef`]s support
/// parsing a strided form which is converted to a semi-affine map automatically.
///
/// ## Strided Layout
///
/// Strided layouts can be expressed using strides to encode the distance, in number of elements, in (linear) memory
/// between successive entries along a particular dimension. For example, a row-major strided layout for
/// `memref<2x3x4xf32>` is `strided<[12, 4, 1]>`, where the last dimension is contiguous as indicated by the unit
/// stride and the remaining strides are products of the sizes of faster-varying dimensions. Strided layout can also
/// express non-contiguity. For example, `memref<2x3, strided<[6, 2]>>` only accesses even elements of the dense
/// consecutive storage along the innermost dimension.
///
/// Strided layouts support an optional offset that indicates the distance, in the number of elements, between the
/// beginning of the memory reference and the first accessed element. When omitted, the offset is considered to be
/// zero. That is, `memref<2, strided<[2], offset: 0>>` and `memref<2, strided<[2]>>` are strictly the same type.
///
/// Both offsets and strides may be dynamic (i.e., unknown at compile time). This is represented by using a question
/// mark (`?`) instead of a value in the [`Display`](std::fmt::Display) rendering of the [`MemRefTypeRef`]
/// (and of the layout itself).
///
/// Strided layouts convert into the following canonical one-dimensional affine form through explicit linearization:
///
/// ```text
/// affine_map<(d0, ... dN)[offset, stride0, ... strideN] -> (offset + d0 * stride0 + ... dN * strideN)>
/// ```
///
/// Therefore, it is never subject to the implicit row-major layout interpretation.
///
/// ## Affine Map Layout
///
/// Affine map layout may be represented directly as an [`AffineMap`] from the index space to the storage space. For
/// example, consider an index map which maps a 2-dimensional index from a 2x2 index space to a 3x3 index space where
/// the 2x2 sub-space is offset by `S0` and `S1` along each dimension: `#example_map (d0, d1) -> (d0 + S0, d1 + S1)`.
///
/// Semi-affine maps are sufficiently flexible to represent a wide variety of dense storage layouts, including row-
/// and column-major layouts, as well as tiled layouts:
///
/// ```text
/// #layout_map_row_major = (i, j) -> (i, j)
/// #layout_map_col_major = (i, j) -> (j, i)
/// #layout_tiled_64x64 = (i, j) -> (i floordiv 64, j floordiv 64, i mod 64, j mod 64)
/// ```
///
/// # Examples
///
/// The following are examples of [`MemRefTypeRef`]s represented using their [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// // Identity index/layout map:
/// #identity = affine_map<(d0, d1) -> (d0, d1)>
///
/// // Column major layout:
///  #col_major = affine_map<(d0, d1, d2) -> (d2, d1, d0)>
///
/// // 2D tiled layout with tiles of size 128 x 256:
/// #tiled_2d_128x256 = affine_map<(d0, d1) -> (d0 div 128, d1 div 256, d0 mod 128, d1 mod 256)>
///
/// // Tiled data layout with non-constant tile sizes:
/// #tiled_dynamic = affine_map<(d0, d1)[s0, s1] -> (d0 floordiv s0, d1 floordiv s1, d0 mod s0, d1 mod s1)>
///
/// // Layout that yields a padding on two at either end of the minor dimension:
/// #padded = affine_map<(d0, d1) -> (d0, (d1 + 2) floordiv 2, (d1 + 2) mod 2)>
///
/// // The dimension list "16x32" defines the following 2D index space:
/// //
/// //   { (i, j) : 0 <= i < 16, 0 <= j < 32 }
/// //
/// memref<16x32xf32, #identity>
///
/// // The dimension list "16x4x?" defines the following 3D index space:
/// //
/// //   { (i, j, k) : 0 <= i < 16, 0 <= j < 4, 0 <= k < N }
/// //
/// // where N is a symbol which represents the runtime value of the size of
/// // the third dimension.
/// //
/// // %N here binds to the size of the third dimension:
/// %A = alloc(%N) : memref<16x4x?xf32, #col_major>
///
/// // 2D dynamic shaped memref that also has a dynamically sized tiled layout. The memref index space is of size
/// // %M x %N, while %B1 and %B2 bind to the symbols s0, s1 respectively of the layout map #tiled_dynamic.
/// // Data tiles of size %B1 x %B2 in the logical space will be stored contiguously in memory. The allocation size
/// // will be (%M ceildiv %B1) * %B1 * (%N ceildiv %B2) * %B2 f32 elements.
/// %T = alloc(%M, %N) [%B1, %B2] : memref<?x?xf32, #tiled_dynamic>
///
/// // Memory reference that has a two-element padding at either end. The allocation size will fit 16 * 64 float
/// // elements of data.
/// %P = alloc() : memref<16x64xf32, #padded>
///
/// // Affine map with symbol 's0' used as offset for the first dimension:
/// #imapS = affine_map<(d0, d1) [s0] -> (d0 + s0, d1)>
/// // Allocate memref and bind the following symbols:
/// // '%n' is bound to the dynamic second dimension of the memref type.
/// // '%o' is bound to the symbol 's0' in the affine map of the memref type.
/// %n = ...
/// %o = ...
/// %A = alloc (%n)[%o] : <16x?xf32, #imapS>
/// ```
///
/// Refer to the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype) for more information.
#[derive(Copy, Clone)]
pub struct MemRefTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> MemRefTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`MemRefTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirMemRefTypeGetTypeID()).unwrap() }
    }

    /// Returns the rank of this [`MemRefTypeRef`] (i.e., the number of dimensions it has).
    pub fn rank(&self) -> usize {
        unsafe { mlirShapedTypeGetRank(self.handle) as usize }
    }

    /// Returns all dimension [`Size`]s of this [`MemRefTypeRef`].
    pub fn dimensions(&self) -> impl Iterator<Item = Size> {
        (0..self.rank()).map(|dimension| self.dimension(dimension))
    }

    /// Returns `true` if all dimensions of this [`MemRefTypeRef`] has a static size.
    pub fn has_static_shape(&self) -> bool {
        self.dimensions().all(|dimension| dimension.is_static())
    }

    /// Returns the `dimension`-th [`Size`] of this [`MemRefTypeRef`].
    ///
    /// Note that this function will panic if the provided dimension is out of bounds.
    pub fn dimension(&self, dimension: usize) -> Size {
        if dimension >= self.rank() {
            panic!("dimension is out of bounds");
        }
        unsafe {
            if mlirShapedTypeIsStaticDim(self.handle, dimension.cast_signed()) {
                Size::Static(mlirShapedTypeGetDimSize(self.handle, dimension.cast_signed()) as usize)
            } else {
                Size::Dynamic
            }
        }
    }

    /// Returns the layout [`Attribute`] of this [`MemRefTypeRef`].
    pub fn layout(&self) -> Option<AttributeRef<'c, 't>> {
        unsafe { AttributeRef::from_c_api(mlirMemRefTypeGetLayout(self.handle), self.context) }
    }

    /// Returns the strides and the offset of this [`MemRefTypeRef`]. The strides are [`Size`]s that encode the distance
    /// (as a number of elements) between successive entries in the underlying [`MemRefTypeRef`] instance, along each
    /// dimension. For example, `memref<42x16xf32, (64 * d0 + d1)>` specifies a view into a non-contiguous memory
    /// region of `42` by `16` [`Float32TypeRef`](crate::Float32TypeRef) elements in which the distance between two
    /// consecutive elements along the outer dimension is `1` and the distance between two consecutive elements along
    /// the inner dimension is `64`.
    ///
    /// The convention is that the strides for dimensions `0` through `N` appear in this order in the list, in order
    /// to make indexing into the result intuitive.
    pub fn strides_and_offset(&self) -> Option<(Vec<Size>, Size)> {
        unsafe {
            let rank = self.rank();
            let mut strides: Vec<i64> = Vec::with_capacity(rank);
            let mut offset = 0i64;
            let result = LogicalResult::from_c_api(mlirMemRefTypeGetStridesAndOffset(
                self.handle,
                strides.as_mut_ptr(),
                &mut offset as *mut i64,
            ));
            if result.is_success() {
                let strides = strides.into_iter().map(|stride| Size::from_c_api(stride)).collect();
                let offset = Size::from_c_api(offset);
                Some((strides, offset))
            } else {
                None
            }
        }
    }

    /// Returns the memory space [`Attribute`] of this [`MemRefTypeRef`].
    pub fn memory_space(&self) -> Option<AttributeRef<'c, 't>> {
        unsafe { AttributeRef::from_c_api(mlirMemRefTypeGetMemorySpace(self.handle), self.context) }
    }

    /// Returns the [`AffineMap`] of this [`MemRefTypeRef`].
    pub fn affine_map(&self) -> AffineMap<'c, 't> {
        unsafe { AffineMap::from_c_api(mlirMemRefTypeGetAffineMap(self.handle), self.context).unwrap() }
    }
}

impl<'c, 't> ShapedType<'c, 't> for MemRefTypeRef<'c, 't> {}

mlir_subtype_trait_impls!(MemRefTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = MemRef);

impl<'t> Context<'t> {
    /// Creates a new [`MemRefTypeRef`] owned by this [`Context`]. If any of the arguments are invalid, then
    /// this function will return [`None`] and will also emit the appropriate diagnostics at the provided location.
    /// For example, this can happen if `element_type` is not an [`IndexTypeRef`](crate::IndexTypeRef),
    /// an [`IntegerTypeRef`](crate::IntegerTypeRef), a [`FloatType`](crate::FloatTypeRef),
    /// a [`VectorTypeRef`](crate::VectorTypeRef), another [`MemRefTypeRef`], or a [`UnrankedMemRefTypeRef`].
    pub fn mem_ref_type<'c, T: Type<'c, 't>, L: Location<'c, 't>>(
        &'c self,
        element_type: T,
        shape: &[Size],
        layout: Option<AttributeRef<'c, 't>>,
        memory_space: Option<AttributeRef<'c, 't>>,
        location: L,
    ) -> Option<MemRefTypeRef<'c, 't>> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe {
            let dimensions = shape.iter().map(|dimension| dimension.to_c_api()).collect::<Vec<_>>();
            MemRefTypeRef::from_c_api(
                mlirMemRefTypeGetChecked(
                    location.to_c_api(),
                    element_type.to_c_api(),
                    dimensions.len().cast_signed(),
                    dimensions.as_ptr(),
                    layout.unwrap_or_else(|| self.null_attribute()).to_c_api(),
                    memory_space.unwrap_or_else(|| self.null_attribute()).to_c_api(),
                ),
                self,
            )
        }
    }

    /// Creates a new contiguous [`MemRefTypeRef`] (i.e., one with a contiguous memory layout) owned by this
    /// [`Context`]. If any of the arguments are invalid, then this function will return [`None`] and will also emit
    /// the appropriate diagnostics at the provided location. For example, this can happen if `element_type` is not
    /// an [`IndexTypeRef`](crate::IndexTypeRef), an [`IntegerTypeRef`](crate::IntegerTypeRef),
    /// a [`FloatType`](crate::FloatTypeRef), a [`VectorTypeRef`](crate::VectorTypeRef), another [`MemRefTypeRef`],
    /// or a [`UnrankedMemRefTypeRef`].
    pub fn contiguous_mem_ref_type<'c, T: Type<'c, 't>, L: Location<'c, 't>>(
        &'c self,
        element_type: T,
        shape: &[Size],
        memory_space: Option<AttributeRef<'c, 't>>,
        location: L,
    ) -> Option<MemRefTypeRef<'c, 't>> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe {
            let dimensions = shape.iter().map(|dimension| dimension.to_c_api()).collect::<Vec<_>>();
            MemRefTypeRef::from_c_api(
                mlirMemRefTypeContiguousGetChecked(
                    location.to_c_api(),
                    element_type.to_c_api(),
                    dimensions.len().cast_signed(),
                    dimensions.as_ptr(),
                    memory_space.unwrap_or_else(|| self.null_attribute()).to_c_api(),
                ),
                self,
            )
        }
    }
}

/// Built-in MLIR [`Type`] that represents a shaped reference to a region of memory with an unknown rank (i.e., an
/// unknown number of dimensions). This is in contrast to [`MemRefTypeRef`] which is used to represent shaped references
/// to regions of memory with a known number of dimensions.
///
/// The purpose of [`UnrankedMemRefTypeRef`]s is to allow external library functions to receive arguments of any rank
/// without versioning the functions based on the rank. Other uses of this type are disallowed or will result in
/// undefined behavior.
///
/// # Codegen for [`UnrankedMemRefTypeRef`]s
///
/// Using [`UnrankedMemRefTypeRef`]s in codegen besides the case mentioned above is highly discouraged. Codegen is
/// concerned with generating loop nests and specialized instructions for high-performance. [`UnrankedMemRefTypeRef`]s
/// are concerned with hiding the rank and thus, the number of enclosing loops required to iterate over the data.
/// However, if there is a need to codegen for [`UnrankedMemRefTypeRef`]s, one possible path is to cast into a static
/// ranked type based on the dynamic rank. Another possible path is to emit a single while loop conditioned on a linear
/// index and perform delinearization of the linear index to a dynamic array containing the (unranked) indices. While
/// this is possible, it is expected to not be a good idea to perform this during codegen as the cost of the
/// translations is expected to be prohibitive and optimizations at this level are not expected to be worthwhile.
/// If expressiveness is the main concern, irrespective of performance, passing [`UnrankedMemRefTypeRef`]s to an
/// external C++ library and implementing rank-agnostic logic there is expected to be significantly simpler.
///
/// [`UnrankedMemRefTypeRef`]s may provide expressiveness gains in the future and help bridge the gap with
/// [`UnrankedTensorTypeRef`](crate::UnrankedTensorTypeRef)s. They are not expected to be exposed to codegen but one may
/// query the rank of an [`UnrankedMemRefTypeRef`] (though a special op will be needed for this purpose) and perform a
/// switch and cast to a [`MemRefTypeRef`] as a prerequisite to codegen.
///
/// Refer to the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype) for more information.
#[derive(Copy, Clone)]
pub struct UnrankedMemRefTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> UnrankedMemRefTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`UnrankedMemRefTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirUnrankedMemRefTypeGetTypeID()).unwrap() }
    }

    /// Returns the memory space [`Attribute`] of this [`UnrankedMemRefTypeRef`].
    pub fn memory_space(&self) -> Option<AttributeRef<'c, 't>> {
        unsafe { AttributeRef::from_c_api(mlirUnrankedMemrefGetMemorySpace(self.handle), self.context) }
    }
}

impl<'c, 't> ShapedType<'c, 't> for UnrankedMemRefTypeRef<'c, 't> {}

mlir_subtype_trait_impls!(UnrankedMemRefTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = UnrankedMemRef);

impl<'t> Context<'t> {
    /// Creates a new [`UnrankedMemRefTypeRef`] owned by this [`Context`]. If any of the arguments are invalid, then
    /// this function will return [`None`] and will also emit the appropriate diagnostics at the provided location.
    /// For example, this can happen if `element_type` is not an [`IndexTypeRef`](crate::IndexTypeRef),
    /// an [`IntegerTypeRef`](crate::IntegerTypeRef), a [`FloatType`](crate::FloatTypeRef),
    /// a [`VectorTypeRef`](crate::VectorTypeRef), a [`MemRefTypeRef`], or another [`UnrankedMemRefTypeRef`].
    pub fn unranked_mem_ref_type<'c, T: Type<'c, 't>, L: Location<'c, 't>>(
        &'c self,
        element_type: T,
        memory_space: Option<AttributeRef<'c, 't>>,
        location: L,
    ) -> Option<UnrankedMemRefTypeRef<'c, 't>> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe {
            UnrankedMemRefTypeRef::from_c_api(
                mlirUnrankedMemRefTypeGetChecked(
                    location.to_c_api(),
                    element_type.to_c_api(),
                    memory_space.unwrap_or_else(|| self.null_attribute()).to_c_api(),
                ),
                self,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_mem_ref_type_type_id() {
        let context = Context::new();
        let location = context.unknown_location();
        let mem_ref_type = MemRefTypeRef::type_id();
        let mem_ref_1_type =
            context.mem_ref_type(context.float32_type(), &[Size::Static(32)], None, None, location).unwrap();
        let mem_ref_2_type =
            context.mem_ref_type(context.float32_type(), &[Size::Static(64)], None, None, location).unwrap();
        assert_eq!(mem_ref_1_type.type_id(), mem_ref_2_type.type_id());
        assert_eq!(mem_ref_type, mem_ref_1_type.type_id());
    }

    #[test]
    fn test_mem_ref_type() {
        let context = Context::new();
        let location = context.unknown_location();

        // Non-contiguous mem-ref type with valid element type.
        let element_type = context.bfloat16_type();
        let shape = vec![Size::Static(32), Size::Dynamic, Size::Dynamic, Size::Static(2)];
        let r#type = context.mem_ref_type(element_type, &shape, None, None, location).unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.rank(), 4);
        assert_eq!(r#type.dimensions().collect::<Vec<_>>(), shape);
        assert_eq!(r#type.dimension(0), Size::Static(32));
        assert_eq!(r#type.dimension(1), Size::Dynamic);
        assert_eq!(r#type.dimension(2), Size::Dynamic);
        assert_eq!(r#type.dimension(3), Size::Static(2));
        assert_eq!(r#type.layout(), Some(context.affine_map_attribute(context.identity_affine_map(4)).as_ref()),);
        assert_eq!(r#type.strides_and_offset(), Some((vec![], Size::Static(0))));
        assert_eq!(r#type.memory_space(), None);
        assert_eq!(r#type.affine_map(), context.identity_affine_map(4));
        assert!(!r#type.has_static_shape());
        assert_eq!(r#type.element_type(), element_type);

        // Contiguous mem-ref type with valid element type.
        let r#type = context.contiguous_mem_ref_type(element_type, &shape, None, location).unwrap();
        assert_eq!(r#type.rank(), 4);
        assert_eq!(r#type.dimensions().collect::<Vec<_>>(), shape);
        assert_eq!(r#type.layout(), Some(context.affine_map_attribute(context.identity_affine_map(4)).as_ref()),);
        assert_eq!(r#type.affine_map(), context.identity_affine_map(4));

        // Invalid element type.
        let element_type = context.none_type();
        let r#type = context.mem_ref_type(element_type, &shape, None, None, location);
        assert!(r#type.is_none());
    }

    #[test]
    fn test_mem_ref_type_equality() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.bfloat16_type();
        let shape = vec![Size::Static(32), Size::Dynamic];

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.mem_ref_type(element_type, &shape, None, None, location).unwrap();
        let type_2 = context.mem_ref_type(element_type, &shape, None, None, location).unwrap();
        assert_eq!(type_1, type_2);

        // Different shapes from the same context must not be equal.
        let other_shape = vec![Size::Static(32), Size::Static(16)];
        let type_2 = context.mem_ref_type(element_type, &other_shape, None, None, location).unwrap();
        assert_ne!(type_1, type_2);

        // Non-contiguous and contiguous mem-ref types are equal when no layout is specified.
        let type_1 = context.mem_ref_type(element_type, &shape, None, None, location).unwrap();
        let type_2 = context.contiguous_mem_ref_type(element_type, &shape, None, location).unwrap();
        assert_eq!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let location = context.unknown_location();
        let other_element_type = context.bfloat16_type();
        let type_2 = context.mem_ref_type(other_element_type, &shape, None, None, location).unwrap();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_mem_ref_type_display_and_debug() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.bfloat16_type();
        let shape = vec![Size::Static(32), Size::Dynamic, Size::Dynamic, Size::Static(2)];
        let r#type = context.mem_ref_type(element_type, &shape, None, None, location).unwrap();
        test_type_display_and_debug(r#type, "memref<32x?x?x2xbf16>");
    }

    #[test]
    fn test_mem_ref_type_parsing() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.bfloat16_type();
        let shape = vec![Size::Static(32), Size::Dynamic, Size::Dynamic, Size::Static(2)];
        assert_eq!(
            context.parse_type("memref<32x?x?x2xbf16>").unwrap(),
            context.mem_ref_type(element_type, &shape, None, None, location).unwrap()
        );
    }

    #[test]
    fn test_mem_ref_type_casting() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.bfloat16_type();
        let shape = vec![Size::Static(32), Size::Dynamic];
        let r#type = context.mem_ref_type(element_type, &shape, None, None, location).unwrap();
        test_type_casting(r#type);
    }

    #[test]
    fn test_unranked_mem_ref_type_type_id() {
        let context = Context::new();
        let location = context.unknown_location();
        let unranked_mem_ref_type = UnrankedMemRefTypeRef::type_id();
        let unranked_mem_ref_1_type = context.unranked_mem_ref_type(context.float32_type(), None, location).unwrap();
        let unranked_mem_ref_2_type = context.unranked_mem_ref_type(context.bfloat16_type(), None, location).unwrap();
        assert_eq!(unranked_mem_ref_1_type.type_id(), unranked_mem_ref_2_type.type_id());
        assert_eq!(unranked_mem_ref_type, unranked_mem_ref_1_type.type_id());
    }

    #[test]
    fn test_unranked_mem_ref_type() {
        let context = Context::new();
        let location = context.unknown_location();

        // Valid element type.
        let element_type = context.bfloat16_type();
        let r#type = context.unranked_mem_ref_type(element_type, None, location).unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.element_type(), element_type);
        assert_eq!(r#type.memory_space(), None);

        // Invalid element type.
        let element_type = context.none_type();
        let r#type = context.unranked_mem_ref_type(element_type, None, location);
        assert!(r#type.is_none());
    }

    #[test]
    fn test_unranked_mem_ref_type_equality() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.bfloat16_type();

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.unranked_mem_ref_type(element_type, None, location).unwrap();
        let type_2 = context.unranked_mem_ref_type(element_type, None, location).unwrap();
        assert_eq!(type_1, type_2);

        // Different element types from the same context must not be equal.
        let other_element_type = context.float32_type();
        let type_2 = context.unranked_mem_ref_type(other_element_type, None, location).unwrap();
        assert_ne!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let location = context.unknown_location();
        let other_element_type = context.bfloat16_type();
        let type_2 = context.unranked_mem_ref_type(other_element_type, None, location).unwrap();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_unranked_mem_ref_type_display_and_debug() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.bfloat16_type();
        let r#type = context.unranked_mem_ref_type(element_type, None, location).unwrap();
        test_type_display_and_debug(r#type, "memref<*xbf16>");
    }

    #[test]
    fn test_unranked_mem_ref_type_parsing() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.bfloat16_type();
        assert_eq!(
            context.parse_type("memref<*xbf16>").unwrap(),
            context.unranked_mem_ref_type(element_type, None, location).unwrap()
        );
    }

    #[test]
    fn test_unranked_mem_ref_type_casting() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.bfloat16_type();
        let r#type = context.unranked_mem_ref_type(element_type, None, location).unwrap();
        test_type_casting(r#type);
    }
}
