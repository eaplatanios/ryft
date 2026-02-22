use ryft_xla_sys::bindings::{
    MlirType, mlirShapedTypeGetDimSize, mlirShapedTypeGetRank, mlirVectorTypeGetChecked,
    mlirVectorTypeGetScalableChecked, mlirVectorTypeGetTypeID, mlirVectorTypeIsDimScalable, mlirVectorTypeIsScalable,
};

use crate::{Context, Location, Type, TypeId, mlir_subtype_trait_impls};

use super::ShapedType;

/// Represents the size of a [`VectorTypeRef`] dimension. Refer to the documentation of [`VectorTypeRef`] for what
/// kinds of dimensions sizes the different variants of this enum represent.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VectorTypeDimension {
    Fixed(usize),
    Scalable(usize),
}

/// Built-in MLIR [`Type`] that represents a multidimensional SIMD vector type.
///
/// This type is used by target-specific operation sets like AVX or SVE. While the most common use is for 1-D vectors
/// (e.g., `vector<16 x f32>`), MLIR also supports multidimensional registers on targets that support them (e.g., on
/// TPUs). The dimensions of a vector type can be fixed-length, scalable-length, or a combination of the two. Scalable
/// dimensions are defined as dimensions whose size is dynamic, but is always a multiple of a statically-known scalar
/// factor. Scalable dimensions in the string rendering of a vector type are rendered as that scalar factor surrounded
/// by square brackets.
///
/// # Examples
///
/// The following are examples of [`VectorTypeRef`]s represented using their [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// vector<3x42xi32>     => 2D fixed-length vector with 3*42=126 i32 elements.
/// vector<[4]xf32>      => 1D scalable-length vector with 4*N f32 elements.
/// vector<[2]x[8]xf32>  => 2D scalable-length vector with 2*8*N=16*N f32 elements.
/// vector<4x[4]xf32>    => 2D mixed-length vector with 4 scalable vectors with 4*N f32 elements each.
/// vector<2x[4]x8xf32>  => 3D mixed-length vector in which only the middle dimension is scalable.
/// ```
///
/// Refer to the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#vectortype) for more information.
#[derive(Copy, Clone)]
pub struct VectorTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> VectorTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`VectorTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirVectorTypeGetTypeID()).unwrap() }
    }

    /// Returns the rank of this [`VectorTypeRef`] (i.e., the number of dimensions it has).
    pub fn rank(&self) -> usize {
        unsafe { mlirShapedTypeGetRank(self.to_c_api()) as usize }
    }

    /// Returns all dimension [`VectorTypeDimension`]s of this [`VectorTypeRef`].
    pub fn dimensions(&self) -> impl Iterator<Item = VectorTypeDimension> {
        (0..self.rank()).map(|dimension| self.dimension(dimension))
    }

    /// Returns the `dimension`-th [`VectorTypeDimension`] of this [`VectorTypeRef`].
    ///
    /// Note that this function will panic if the provided dimension is out of bounds.
    pub fn dimension(&self, dimension: usize) -> VectorTypeDimension {
        if dimension >= self.rank() {
            panic!("dimension is out of bounds");
        }
        let size = unsafe { mlirShapedTypeGetDimSize(self.to_c_api(), dimension.cast_signed()) as usize };
        if self.is_dimension_scalable(dimension) {
            VectorTypeDimension::Scalable(size)
        } else {
            VectorTypeDimension::Fixed(size)
        }
    }

    /// Returns `true` if this [`VectorTypeRef`] is scalable (i.e., has at least one scalable dimension),
    /// and `false` otherwise.
    pub fn is_scalable(&self) -> bool {
        unsafe { mlirVectorTypeIsScalable(self.handle) }
    }

    /// Returns `true` if the `dimension`-th dimension of this [`VectorTypeRef`] is scalable, and `false` otherwise.
    ///
    /// Note that this function will panic if the provided dimension is out of bounds.
    pub fn is_dimension_scalable(&self, dimension: usize) -> bool {
        if dimension >= self.rank() {
            panic!("dimension is out of bounds");
        }
        unsafe { mlirVectorTypeIsDimScalable(self.handle, dimension.cast_signed()) }
    }
}

impl<'c, 't> ShapedType<'c, 't> for VectorTypeRef<'c, 't> {}

mlir_subtype_trait_impls!(VectorTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = Vector);

impl<'t> Context<'t> {
    /// Creates a new [`VectorTypeRef`] owned by this [`Context`]. If any of the arguments are invalid, then this
    /// function will return [`None`] and will also emit the appropriate diagnostics at the provided location. For
    /// example, this can happen if `element_type` is not an [`IndexTypeRef`](crate::IndexTypeRef), an
    /// [`IntegerTypeRef`](crate::IntegerTypeRef), or a [`FloatType`](crate::FloatTypeRef).
    pub fn vector_type<'c, T: Type<'c, 't>, L: Location<'c, 't>>(
        &'c self,
        element_type: T,
        shape: &[VectorTypeDimension],
        location: L,
    ) -> Option<VectorTypeRef<'c, 't>> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        let dimensions = shape
            .iter()
            .map(|dimension| match dimension {
                VectorTypeDimension::Fixed(size) => *size as i64,
                VectorTypeDimension::Scalable(size) => *size as i64,
            })
            .collect::<Vec<_>>();
        let scalable = shape
            .iter()
            .map(|dimension| matches!(dimension, VectorTypeDimension::Scalable(_)))
            .collect::<Vec<_>>();
        let is_scalable = scalable.iter().any(|is_scalable| *is_scalable);
        unsafe {
            let handle = if is_scalable {
                mlirVectorTypeGetScalableChecked(
                    location.to_c_api(),
                    dimensions.len().cast_signed(),
                    dimensions.as_ptr(),
                    scalable.as_ptr(),
                    element_type.to_c_api(),
                )
            } else {
                mlirVectorTypeGetChecked(
                    location.to_c_api(),
                    dimensions.len().cast_signed(),
                    dimensions.as_ptr(),
                    element_type.to_c_api(),
                )
            };
            VectorTypeRef::from_c_api(handle, &self)
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_vector_type_type_id() {
        let context = Context::new();
        let location = context.unknown_location();
        let vector_type = VectorTypeRef::type_id();
        let vector_type_1 = context.vector_type(context.float32_type(), &[VectorTypeDimension::Fixed(16)], location);
        let vector_type_2 = context.vector_type(context.float32_type(), &[VectorTypeDimension::Fixed(32)], location);
        assert_eq!(vector_type_1.unwrap().type_id(), vector_type_2.unwrap().type_id());
        assert_eq!(vector_type, vector_type_1.unwrap().type_id());
    }

    #[test]
    fn test_vector_type() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.float32_type();

        // Fixed dimension.
        let shape = vec![VectorTypeDimension::Fixed(16)];
        let r#type = context.vector_type(element_type, &shape, location).unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.rank(), 1);
        assert_eq!(r#type.dimensions().collect::<Vec<_>>(), shape);
        assert_eq!(r#type.dimension(0), VectorTypeDimension::Fixed(16));
        assert!(!r#type.is_scalable());
        assert!(!r#type.is_dimension_scalable(0));
        assert_eq!(r#type.element_type(), element_type);

        // Scalable dimension.
        let shape = vec![VectorTypeDimension::Scalable(4)];
        let r#type = context.vector_type(element_type, &shape, location).unwrap();
        assert_eq!(r#type.rank(), 1);
        assert_eq!(r#type.dimension(0), VectorTypeDimension::Scalable(4));
        assert!(r#type.is_scalable());
        assert!(r#type.is_dimension_scalable(0));

        // Mixed dimensions.
        let shape = vec![
            VectorTypeDimension::Fixed(32),
            VectorTypeDimension::Scalable(4),
            VectorTypeDimension::Scalable(2),
            VectorTypeDimension::Fixed(2),
        ];
        let r#type = context.vector_type(element_type, &shape, location).unwrap();
        assert_eq!(r#type.rank(), 4);
        assert_eq!(r#type.dimensions().collect::<Vec<_>>(), shape);
        assert_eq!(r#type.dimension(0), VectorTypeDimension::Fixed(32));
        assert_eq!(r#type.dimension(1), VectorTypeDimension::Scalable(4));
        assert_eq!(r#type.dimension(2), VectorTypeDimension::Scalable(2));
        assert_eq!(r#type.dimension(3), VectorTypeDimension::Fixed(2));
        assert!(r#type.is_scalable());
        assert!(!r#type.is_dimension_scalable(0));
        assert!(r#type.is_dimension_scalable(1));
        assert!(r#type.is_dimension_scalable(2));
        assert!(!r#type.is_dimension_scalable(3));

        // Invalid element type.
        let shape = vec![VectorTypeDimension::Fixed(16)];
        let r#type = context.vector_type(context.none_type(), &shape, location);
        assert!(r#type.is_none());
    }

    #[test]
    fn test_vector_type_equality() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.float32_type();
        let shape = vec![VectorTypeDimension::Fixed(16)];

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.vector_type(element_type, &shape, location);
        let type_2 = context.vector_type(element_type, &shape, location);
        assert_eq!(type_1, type_2);

        // Different shapes from the same context must not be equal.
        let other_shape = vec![VectorTypeDimension::Fixed(32)];
        let type_2 = context.vector_type(element_type, &other_shape, location);
        assert_ne!(type_1, type_2);

        // Fixed vs scalable dimensions must not be equal.
        let scalable_shape = vec![VectorTypeDimension::Scalable(16)];
        let type_2 = context.vector_type(element_type, &scalable_shape, location);
        assert_ne!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.float32_type();
        let type_2 = context.vector_type(element_type, &shape, location);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_vector_type_display_and_debug() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.float32_type();

        let shape = vec![VectorTypeDimension::Fixed(16)];
        let r#type = context.vector_type(element_type, &shape, location).unwrap();
        test_type_display_and_debug(r#type, "vector<16xf32>");

        let shape = vec![VectorTypeDimension::Scalable(4)];
        let r#type = context.vector_type(element_type, &shape, location).unwrap();
        test_type_display_and_debug(r#type, "vector<[4]xf32>");

        let shape = vec![
            VectorTypeDimension::Fixed(32),
            VectorTypeDimension::Scalable(4),
            VectorTypeDimension::Scalable(2),
            VectorTypeDimension::Fixed(2),
        ];
        let r#type = context.vector_type(element_type, &shape, location).unwrap();
        test_type_display_and_debug(r#type, "vector<32x[4]x[2]x2xf32>");
    }

    #[test]
    fn test_vector_type_parsing() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.float32_type();

        let shape = vec![VectorTypeDimension::Fixed(16)];
        assert_eq!(
            context.parse_type("vector<16xf32>").unwrap(),
            context.vector_type(element_type, &shape, location).unwrap()
        );

        let shape = vec![VectorTypeDimension::Scalable(4)];
        assert_eq!(
            context.parse_type("vector<[4]xf32>").unwrap(),
            context.vector_type(element_type, &shape, location).unwrap()
        );

        let shape = vec![
            VectorTypeDimension::Fixed(32),
            VectorTypeDimension::Scalable(4),
            VectorTypeDimension::Scalable(2),
            VectorTypeDimension::Fixed(2),
        ];
        assert_eq!(
            context.parse_type("vector<32x[4]x[2]x2xf32>").unwrap(),
            context.vector_type(element_type, &shape, location).unwrap()
        );
    }

    #[test]
    fn test_vector_type_casting() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.float32_type();
        let shape = vec![VectorTypeDimension::Fixed(16)];
        let r#type = context.vector_type(element_type, &shape, location).unwrap();
        test_type_casting(r#type);
    }
}
