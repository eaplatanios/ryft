use ryft_xla_sys::bindings::MlirType;
use ryft_xla_sys::mlir::dialects::sparse_tensor::{
    mlirSparseTensorIterSpaceTypeGet, mlirSparseTensorIterSpaceTypeGetEncoding,
    mlirSparseTensorIterSpaceTypeGetLowerLevel, mlirSparseTensorIterSpaceTypeGetUpperLevel,
    mlirSparseTensorIteratorTypeGet, mlirSparseTensorIteratorTypeGetEncoding,
    mlirSparseTensorIteratorTypeGetLowerLevel, mlirSparseTensorIteratorTypeGetUpperLevel,
    mlirSparseTensorStorageSpecifierTypeGet, mlirSparseTensorStorageSpecifierTypeGetEncoding,
    mlirTypeIsASparseTensorIterSpaceType, mlirTypeIsASparseTensorIteratorType,
    mlirTypeIsASparseTensorStorageSpecifierType,
};

use crate::{Attribute, Context, DialectHandle, Error, Type, mlir_subtype_trait_impls};

use super::SparseTensorEncodingAttributeRef;

/// Sparse tensor storage specifier [`Type`]. Values of this type aggregate low-level sparse storage metadata for a
/// specific sparse tensor encoding.
#[derive(Copy, Clone)]
pub struct StorageSpecifierTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> StorageSpecifierTypeRef<'c, 't> {
    /// Returns the sparse tensor encoding associated with this storage specifier.
    pub fn encoding(&self) -> Result<SparseTensorEncodingAttributeRef<'c, 't>, Error> {
        unsafe {
            SparseTensorEncodingAttributeRef::from_c_api(
                mlirSparseTensorStorageSpecifierTypeGetEncoding(self.handle),
                self.context,
            )
            .map_err(|_| Error::internal("invalid sparse tensor storage specifier encoding"))
        }
    }
}

impl<'c, 't> Type<'c, 't> for StorageSpecifierTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsASparseTensorStorageSpecifierType(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR type handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(StorageSpecifierTypeRef<'c, 't> as Type, mlir_type = Type);

/// Sparse tensor iteration-space [`Type`]. It represents the coordinates for stored elements between a consecutive
/// range of storage levels.
#[derive(Copy, Clone)]
pub struct IterationSpaceTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> IterationSpaceTypeRef<'c, 't> {
    /// Returns the sparse tensor encoding associated with this iteration space.
    pub fn encoding(&self) -> Result<SparseTensorEncodingAttributeRef<'c, 't>, Error> {
        unsafe {
            SparseTensorEncodingAttributeRef::from_c_api(
                mlirSparseTensorIterSpaceTypeGetEncoding(self.handle),
                self.context,
            )
            .map_err(|_| Error::internal("invalid sparse tensor iteration space encoding"))
        }
    }

    /// Returns the first storage level covered by this iteration space.
    pub fn lower_level(&self) -> u64 {
        unsafe { mlirSparseTensorIterSpaceTypeGetLowerLevel(self.handle) }
    }

    /// Returns the exclusive upper storage level covered by this iteration space.
    pub fn upper_level(&self) -> u64 {
        unsafe { mlirSparseTensorIterSpaceTypeGetUpperLevel(self.handle) }
    }

    /// Returns the number of storage levels covered by this iteration space.
    pub fn space_dimension(&self) -> u64 {
        self.upper_level() - self.lower_level()
    }
}

impl<'c, 't> Type<'c, 't> for IterationSpaceTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsASparseTensorIterSpaceType(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR type handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(IterationSpaceTypeRef<'c, 't> as Type, mlir_type = Type);

/// Sparse tensor iterator [`Type`]. It points into a matching [`IterationSpaceTypeRef`].
#[derive(Copy, Clone)]
pub struct IteratorTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> IteratorTypeRef<'c, 't> {
    /// Returns the sparse tensor encoding associated with this iterator.
    pub fn encoding(&self) -> Result<SparseTensorEncodingAttributeRef<'c, 't>, Error> {
        unsafe {
            SparseTensorEncodingAttributeRef::from_c_api(
                mlirSparseTensorIteratorTypeGetEncoding(self.handle),
                self.context,
            )
            .map_err(|_| Error::internal("invalid sparse tensor iterator encoding"))
        }
    }

    /// Returns the first storage level covered by this iterator.
    pub fn lower_level(&self) -> u64 {
        unsafe { mlirSparseTensorIteratorTypeGetLowerLevel(self.handle) }
    }

    /// Returns the exclusive upper storage level covered by this iterator.
    pub fn upper_level(&self) -> u64 {
        unsafe { mlirSparseTensorIteratorTypeGetUpperLevel(self.handle) }
    }

    /// Returns the number of storage levels covered by this iterator.
    pub fn space_dimension(&self) -> u64 {
        self.upper_level() - self.lower_level()
    }
}

impl<'c, 't> Type<'c, 't> for IteratorTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsASparseTensorIteratorType(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR type handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(IteratorTypeRef<'c, 't> as Type, mlir_type = Type);

impl<'t> Context<'t> {
    /// Creates a new sparse tensor storage specifier type owned by this [`Context`].
    pub fn sparse_tensor_storage_specifier_type<'c>(
        &'c self,
        encoding: SparseTensorEncodingAttributeRef<'c, 't>,
    ) -> Result<StorageSpecifierTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::sparse_tensor()?)?;
        unsafe {
            StorageSpecifierTypeRef::from_c_api(
                mlirSparseTensorStorageSpecifierTypeGet(*self.handle.borrow(), encoding.to_c_api()),
                self,
            )
            .map_err(|_| Error::internal("invalid sparse tensor storage specifier type"))
        }
    }

    /// Creates a new sparse tensor iteration-space type owned by this [`Context`].
    pub fn sparse_tensor_iteration_space_type<'c>(
        &'c self,
        encoding: SparseTensorEncodingAttributeRef<'c, 't>,
        lower_level: u64,
        upper_level: u64,
    ) -> Result<IterationSpaceTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::sparse_tensor()?)?;
        unsafe {
            IterationSpaceTypeRef::from_c_api(
                mlirSparseTensorIterSpaceTypeGet(*self.handle.borrow(), encoding.to_c_api(), lower_level, upper_level),
                self,
            )
            .map_err(|_| Error::internal("invalid sparse tensor iteration space type"))
        }
    }

    /// Creates a new sparse tensor iterator type owned by this [`Context`].
    pub fn sparse_tensor_iterator_type<'c>(
        &'c self,
        encoding: SparseTensorEncodingAttributeRef<'c, 't>,
        lower_level: u64,
        upper_level: u64,
    ) -> Result<IteratorTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::sparse_tensor()?)?;
        unsafe {
            IteratorTypeRef::from_c_api(
                mlirSparseTensorIteratorTypeGet(*self.handle.borrow(), encoding.to_c_api(), lower_level, upper_level),
                self,
            )
            .map_err(|_| Error::internal("invalid sparse tensor iterator type"))
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::dialects::sparse_tensor::{LevelFormat, LevelType};
    use crate::types::tests::{test_type_casting, test_type_display_and_debug};
    use crate::{Context, Type};

    use super::*;

    #[test]
    fn test_storage_specifier_type() {
        let context = Context::new();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let r#type = context.sparse_tensor_storage_specifier_type(encoding).unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.encoding().unwrap().to_string(), encoding.to_string());
    }

    #[test]
    fn test_storage_specifier_type_equality() {
        let context = Context::new();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let type_1 = context.sparse_tensor_storage_specifier_type(encoding).unwrap();
        let type_2 = context.sparse_tensor_storage_specifier_type(encoding).unwrap();
        assert_eq!(type_1, type_2);

        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Dense)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let type_2 = context.sparse_tensor_storage_specifier_type(encoding).unwrap();
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let type_2 = context
            .sparse_tensor_storage_specifier_type(
                context
                    .sparse_tensor_encoding_attribute(
                        &[LevelType::from(LevelFormat::Compressed)],
                        Some(context.identity_affine_map(1)),
                        None,
                        0,
                        0,
                        None,
                        None,
                        &[],
                    )
                    .unwrap(),
            )
            .unwrap();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_storage_specifier_type_display_and_debug() {
        let context = Context::new();
        let r#type = context
            .sparse_tensor_storage_specifier_type(
                context
                    .sparse_tensor_encoding_attribute(
                        &[LevelType::from(LevelFormat::Compressed)],
                        Some(context.identity_affine_map(1)),
                        None,
                        0,
                        0,
                        None,
                        None,
                        &[],
                    )
                    .unwrap(),
            )
            .unwrap();
        test_type_display_and_debug(
            r#type,
            "!sparse_tensor.storage_specifier<#sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>",
        );
    }

    #[test]
    fn test_storage_specifier_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::sparse_tensor().unwrap()).unwrap();
        let r#type = context
            .sparse_tensor_storage_specifier_type(
                context
                    .sparse_tensor_encoding_attribute(
                        &[LevelType::from(LevelFormat::Compressed)],
                        Some(context.identity_affine_map(1)),
                        None,
                        0,
                        0,
                        None,
                        None,
                        &[],
                    )
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            context
                .parse_type(
                    "!sparse_tensor.storage_specifier<#sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>",
                )
                .unwrap()
                .cast::<StorageSpecifierTypeRef>()
                .unwrap(),
            r#type,
        );
    }

    #[test]
    fn test_storage_specifier_type_casting() {
        let context = Context::new();
        let r#type = context
            .sparse_tensor_storage_specifier_type(
                context
                    .sparse_tensor_encoding_attribute(
                        &[LevelType::from(LevelFormat::Compressed)],
                        Some(context.identity_affine_map(1)),
                        None,
                        0,
                        0,
                        None,
                        None,
                        &[],
                    )
                    .unwrap(),
            )
            .unwrap();
        test_type_casting(r#type);
    }

    #[test]
    fn test_iteration_space_type() {
        let context = Context::new();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let r#type = context.sparse_tensor_iteration_space_type(encoding, 0, 1).unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.encoding().unwrap(), encoding);
        assert_eq!(r#type.lower_level(), 0);
        assert_eq!(r#type.upper_level(), 1);
        assert_eq!(r#type.space_dimension(), 1);
    }

    #[test]
    fn test_iteration_space_type_equality() {
        let context = Context::new();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let type_1 = context.sparse_tensor_iteration_space_type(encoding, 0, 1).unwrap();
        let type_2 = context.sparse_tensor_iteration_space_type(encoding, 0, 1).unwrap();
        assert_eq!(type_1, type_2);

        let type_2 = context.sparse_tensor_iteration_space_type(encoding, 0, 2).unwrap();
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let type_2 = context
            .sparse_tensor_iteration_space_type(
                context
                    .sparse_tensor_encoding_attribute(
                        &[LevelType::from(LevelFormat::Compressed)],
                        Some(context.identity_affine_map(1)),
                        None,
                        0,
                        0,
                        None,
                        None,
                        &[],
                    )
                    .unwrap(),
                0,
                1,
            )
            .unwrap();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_iteration_space_type_display_and_debug() {
        let context = Context::new();
        let r#type = context
            .sparse_tensor_iteration_space_type(
                context
                    .sparse_tensor_encoding_attribute(
                        &[LevelType::from(LevelFormat::Compressed)],
                        Some(context.identity_affine_map(1)),
                        None,
                        0,
                        0,
                        None,
                        None,
                        &[],
                    )
                    .unwrap(),
                0,
                1,
            )
            .unwrap();
        test_type_display_and_debug(
            r#type,
            "!sparse_tensor.iter_space<<{ map = (d0) -> (d0 : compressed) }>, lvls = 0>",
        );
    }

    #[test]
    fn test_iteration_space_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::sparse_tensor().unwrap()).unwrap();
        let r#type = context
            .sparse_tensor_iteration_space_type(
                context
                    .sparse_tensor_encoding_attribute(
                        &[LevelType::from(LevelFormat::Compressed)],
                        Some(context.identity_affine_map(1)),
                        None,
                        0,
                        0,
                        None,
                        None,
                        &[],
                    )
                    .unwrap(),
                0,
                1,
            )
            .unwrap();
        assert_eq!(
            context
                .parse_type(
                    "!sparse_tensor.iter_space<#sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>, lvls = 0>",
                )
                .unwrap()
                .cast::<IterationSpaceTypeRef>()
                .unwrap(),
            r#type,
        );
    }

    #[test]
    fn test_iteration_space_type_casting() {
        let context = Context::new();
        let r#type = context
            .sparse_tensor_iteration_space_type(
                context
                    .sparse_tensor_encoding_attribute(
                        &[LevelType::from(LevelFormat::Compressed)],
                        Some(context.identity_affine_map(1)),
                        None,
                        0,
                        0,
                        None,
                        None,
                        &[],
                    )
                    .unwrap(),
                0,
                1,
            )
            .unwrap();
        test_type_casting(r#type);
    }

    #[test]
    fn test_iterator_type() {
        let context = Context::new();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let r#type = context.sparse_tensor_iterator_type(encoding, 0, 1).unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.encoding().unwrap(), encoding);
        assert_eq!(r#type.lower_level(), 0);
        assert_eq!(r#type.upper_level(), 1);
        assert_eq!(r#type.space_dimension(), 1);
    }

    #[test]
    fn test_iterator_type_equality() {
        let context = Context::new();
        let encoding = context
            .sparse_tensor_encoding_attribute(
                &[LevelType::from(LevelFormat::Compressed)],
                Some(context.identity_affine_map(1)),
                None,
                0,
                0,
                None,
                None,
                &[],
            )
            .unwrap();
        let type_1 = context.sparse_tensor_iterator_type(encoding, 0, 1).unwrap();
        let type_2 = context.sparse_tensor_iterator_type(encoding, 0, 1).unwrap();
        assert_eq!(type_1, type_2);

        let type_2 = context.sparse_tensor_iterator_type(encoding, 0, 2).unwrap();
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let type_2 = context
            .sparse_tensor_iterator_type(
                context
                    .sparse_tensor_encoding_attribute(
                        &[LevelType::from(LevelFormat::Compressed)],
                        Some(context.identity_affine_map(1)),
                        None,
                        0,
                        0,
                        None,
                        None,
                        &[],
                    )
                    .unwrap(),
                0,
                1,
            )
            .unwrap();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_iterator_type_display_and_debug() {
        let context = Context::new();
        let r#type = context
            .sparse_tensor_iterator_type(
                context
                    .sparse_tensor_encoding_attribute(
                        &[LevelType::from(LevelFormat::Compressed)],
                        Some(context.identity_affine_map(1)),
                        None,
                        0,
                        0,
                        None,
                        None,
                        &[],
                    )
                    .unwrap(),
                0,
                1,
            )
            .unwrap();
        test_type_display_and_debug(r#type, "!sparse_tensor.iterator<<{ map = (d0) -> (d0 : compressed) }>, lvls = 0>");
    }

    #[test]
    fn test_iterator_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::sparse_tensor().unwrap()).unwrap();
        let r#type = context
            .sparse_tensor_iterator_type(
                context
                    .sparse_tensor_encoding_attribute(
                        &[LevelType::from(LevelFormat::Compressed)],
                        Some(context.identity_affine_map(1)),
                        None,
                        0,
                        0,
                        None,
                        None,
                        &[],
                    )
                    .unwrap(),
                0,
                1,
            )
            .unwrap();
        assert_eq!(
            context
                .parse_type(
                    "!sparse_tensor.iterator<#sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>, lvls = 0>",
                )
                .unwrap()
                .cast::<IteratorTypeRef>()
                .unwrap(),
            r#type,
        );
    }

    #[test]
    fn test_iterator_type_casting() {
        let context = Context::new();
        let r#type = context
            .sparse_tensor_iterator_type(
                context
                    .sparse_tensor_encoding_attribute(
                        &[LevelType::from(LevelFormat::Compressed)],
                        Some(context.identity_affine_map(1)),
                        None,
                        0,
                        0,
                        None,
                        None,
                        &[],
                    )
                    .unwrap(),
                0,
                1,
            )
            .unwrap();
        test_type_casting(r#type);
    }
}
