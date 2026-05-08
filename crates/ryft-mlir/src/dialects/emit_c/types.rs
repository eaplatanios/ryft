use ryft_xla_sys::bindings::{
    MlirType, mlirEmitCArrayTypeGet, mlirEmitCArrayTypeGetTypeID, mlirEmitCLValueTypeGet, mlirEmitCLValueTypeGetTypeID,
    mlirEmitCOpaqueTypeGet, mlirEmitCOpaqueTypeGetTypeID, mlirEmitCPointerTypeGet, mlirEmitCPointerTypeGetTypeID,
    mlirEmitCPtrDiffTTypeGet, mlirEmitCPtrDiffTTypeGetTypeID, mlirEmitCSignedSizeTTypeGet,
    mlirEmitCSignedSizeTTypeGetTypeID, mlirEmitCSizeTTypeGet, mlirEmitCSizeTTypeGetTypeID, mlirShapedTypeGetDimSize,
    mlirShapedTypeGetElementType, mlirShapedTypeGetRank, mlirTypeIsAEmitCArrayType, mlirTypeIsAEmitCLValueType,
    mlirTypeIsAEmitCOpaqueType, mlirTypeIsAEmitCPointerType, mlirTypeIsAEmitCPtrDiffTType,
    mlirTypeIsAEmitCSignedSizeTType, mlirTypeIsAEmitCSizeTType,
};

use crate::{Context, DialectHandle, Error, ShapedType, StringRef, Type, TypeId, TypeRef, mlir_subtype_trait_impls};

/// Emit-C array [`Type`].
///
/// Emit-C arrays are ranked, statically shaped arrays whose element type is supported by Emit-C.
/// They are emitted as C/C++ array types such as `int32_t[10]`.
///
/// Refer to the [official MLIR Emit-C dialect documentation](https://mlir.llvm.org/docs/Dialects/emitc/#types)
/// for more information.
#[derive(Copy, Clone)]
pub struct ArrayTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl ArrayTypeRef<'_, '_> {
    /// Gets the [`TypeId`] that corresponds to [`ArrayTypeRef`].
    pub fn type_id() -> Result<TypeId<'static>, Error> {
        unsafe { TypeId::from_c_api(mlirEmitCArrayTypeGetTypeID()) }
    }

    /// Returns the rank of this array type.
    pub fn rank(&self) -> Result<usize, Error> {
        usize::try_from(unsafe { mlirShapedTypeGetRank(self.handle) })
            .map_err(|_| Error::internal("invalid EmitC array rank"))
    }

    /// Returns the shape of this array type.
    pub fn shape(&self) -> Result<Vec<usize>, Error> {
        (0..self.rank()?)
            .map(|dimension| {
                usize::try_from(unsafe { mlirShapedTypeGetDimSize(self.handle, dimension.cast_signed()) })
                    .map_err(|_| Error::internal("invalid EmitC array dimension"))
            })
            .collect()
    }

    /// Returns the `dimension`-th static dimension of this array type, or an error if `dimension` is out of bounds.
    pub fn dimension(&self, dimension: usize) -> Result<usize, Error> {
        let rank = self.rank()?;
        if dimension >= rank {
            return Err(Error::invalid_argument(format!(
                "emitc array type dimension {dimension} is out of bounds for rank {rank}"
            )));
        }
        Ok(usize::try_from(unsafe { mlirShapedTypeGetDimSize(self.handle, dimension.cast_signed()) })
            .map_err(|_| Error::internal("invalid EmitC array dimension"))?)
    }

    /// Returns the element [`Type`] of this array type.
    pub fn element_type(&self) -> Result<TypeRef<'_, '_>, Error> {
        unsafe {
            TypeRef::from_c_api(mlirShapedTypeGetElementType(self.handle), self.context)
                .map_err(|_| Error::internal("invalid EmitC array element type"))
        }
    }
}

impl<'c, 't> Type<'c, 't> for ArrayTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if handle.ptr.is_null() {
            return Err(Error::internal("expected non-null MLIR type handle"));
        }
        if unsafe { mlirTypeIsAEmitCArrayType(handle) } {
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

impl<'c, 't> ShapedType<'c, 't> for ArrayTypeRef<'c, 't> {}

mlir_subtype_trait_impls!(ArrayTypeRef<'c, 't> as Type, mlir_type = Type);

/// Emit-C lvalue [`Type`].
///
/// Values of this type can be assigned to and can have their address taken.
///
/// Refer to the [official MLIR Emit-C dialect documentation](https://mlir.llvm.org/docs/Dialects/emitc/#types)
/// for more information.
#[derive(Copy, Clone)]
pub struct LValueTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl LValueTypeRef<'_, '_> {
    /// Gets the [`TypeId`] that corresponds to [`LValueTypeRef`].
    pub fn type_id() -> Result<TypeId<'static>, Error> {
        unsafe { TypeId::from_c_api(mlirEmitCLValueTypeGetTypeID()) }
    }
}

impl<'c, 't> Type<'c, 't> for LValueTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if handle.ptr.is_null() {
            return Err(Error::internal("expected non-null MLIR type handle"));
        }
        if unsafe { mlirTypeIsAEmitCLValueType(handle) } {
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

mlir_subtype_trait_impls!(LValueTypeRef<'c, 't> as Type, mlir_type = Type);

/// Emit-C opaque [`Type`] containing a C/C++ source-level type spelling.
///
/// Refer to the [official MLIR Emit-C dialect documentation](https://mlir.llvm.org/docs/Dialects/emitc/#types)
/// for more information.
#[derive(Copy, Clone)]
pub struct OpaqueTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl OpaqueTypeRef<'_, '_> {
    /// Gets the [`TypeId`] that corresponds to [`OpaqueTypeRef`].
    pub fn type_id() -> Result<TypeId<'static>, Error> {
        unsafe { TypeId::from_c_api(mlirEmitCOpaqueTypeGetTypeID()) }
    }
}

impl<'c, 't> Type<'c, 't> for OpaqueTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if handle.ptr.is_null() {
            return Err(Error::internal("expected non-null MLIR type handle"));
        }
        if unsafe { mlirTypeIsAEmitCOpaqueType(handle) } {
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

mlir_subtype_trait_impls!(OpaqueTypeRef<'c, 't> as Type, mlir_type = Type);

/// Emit-C pointer [`Type`].
///
/// Refer to the [official MLIR Emit-C dialect documentation](https://mlir.llvm.org/docs/Dialects/emitc/#types)
/// for more information.
#[derive(Copy, Clone)]
pub struct PointerTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl PointerTypeRef<'_, '_> {
    /// Gets the [`TypeId`] that corresponds to [`PointerTypeRef`].
    pub fn type_id() -> Result<TypeId<'static>, Error> {
        unsafe { TypeId::from_c_api(mlirEmitCPointerTypeGetTypeID()) }
    }
}

impl<'c, 't> Type<'c, 't> for PointerTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if handle.ptr.is_null() {
            return Err(Error::internal("expected non-null MLIR type handle"));
        }
        if unsafe { mlirTypeIsAEmitCPointerType(handle) } {
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

mlir_subtype_trait_impls!(PointerTypeRef<'c, 't> as Type, mlir_type = Type);

macro_rules! emit_c_context_type {
    ($name:ident, $method:ident, $get_type_id:ident, $is_a:ident, $get:ident, $description:literal $(,)*) => {
        #[doc = "Emit-C "]
        #[doc = $description]
        #[doc = " [`Type`]."]
        ///
        /// Refer to the [official MLIR Emit-C dialect documentation](https://mlir.llvm.org/docs/Dialects/emitc/#types)
        /// for more information.
        #[derive(Copy, Clone)]
        pub struct $name<'c, 't> {
            /// Handle that represents this [`Type`] in the MLIR C API.
            handle: MlirType,

            /// [`Context`] that owns this [`Type`].
            context: &'c Context<'t>,
        }

        impl $name<'_, '_> {
            /// Gets the [`TypeId`] that corresponds to this type.
            pub fn type_id() -> Result<TypeId<'static>, Error> {
                unsafe { TypeId::from_c_api($get_type_id()) }
            }
        }

        impl<'c, 't> Type<'c, 't> for $name<'c, 't> {
            unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
                if handle.ptr.is_null() {
                    return Err(Error::internal("expected non-null MLIR type handle"));
                }
                if unsafe { $is_a(handle) } { Ok(Self { handle, context }) } else { Err(Error::invalid_argument("expected MLIR type handle")) }
            }

            unsafe fn to_c_api(&self) -> MlirType {
                self.handle
            }

            fn context(&self) -> &'c Context<'t> {
                self.context
            }
        }

        mlir_subtype_trait_impls!($name<'c, 't> as Type, mlir_type = Type);
    };
}

emit_c_context_type!(
    SignedSizeTTypeRef,
    emit_c_signed_size_t_type,
    mlirEmitCSignedSizeTTypeGetTypeID,
    mlirTypeIsAEmitCSignedSizeTType,
    mlirEmitCSignedSizeTTypeGet,
    "signed size",
);

emit_c_context_type!(
    PtrDiffTTypeRef,
    emit_c_ptrdiff_t_type,
    mlirEmitCPtrDiffTTypeGetTypeID,
    mlirTypeIsAEmitCPtrDiffTType,
    mlirEmitCPtrDiffTTypeGet,
    "pointer difference",
);

emit_c_context_type!(
    SizeTTypeRef,
    emit_c_size_t_type,
    mlirEmitCSizeTTypeGetTypeID,
    mlirTypeIsAEmitCSizeTType,
    mlirEmitCSizeTTypeGet,
    "size",
);

impl<'t> Context<'t> {
    /// Creates a new Emit-C [`ArrayTypeRef`] owned by this [`Context`].
    pub fn emit_c_array_type<'c, T: Type<'c, 't>>(
        &'c self,
        element_type: T,
        shape: &[usize],
    ) -> Result<ArrayTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::emit_c()?)?;
        let mut shape = shape
            .iter()
            .map(|dimension| {
                i64::try_from(*dimension)
                    .map_err(|_| Error::invalid_argument("invalid arguments to `Context::emit_c_array_type`"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        unsafe {
            ArrayTypeRef::from_c_api(
                mlirEmitCArrayTypeGet(shape.len().cast_signed(), shape.as_mut_ptr(), element_type.to_c_api()),
                self,
            )
            .map_err(|_| Error::invalid_argument("invalid arguments to `Context::emit_c_array_type`"))
        }
    }

    /// Creates a new Emit-C [`LValueTypeRef`] owned by this [`Context`].
    pub fn emit_c_lvalue_type<'c, T: Type<'c, 't>>(&'c self, value_type: T) -> Result<LValueTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::emit_c()?)?;
        unsafe {
            LValueTypeRef::from_c_api(mlirEmitCLValueTypeGet(value_type.to_c_api()), self)
                .map_err(|_| Error::invalid_argument("invalid arguments to `Context::emit_c_lvalue_type`"))
        }
    }

    /// Creates a new Emit-C [`OpaqueTypeRef`] owned by this [`Context`].
    pub fn emit_c_opaque_type<'c, S: AsRef<str>>(&'c self, value: S) -> Result<OpaqueTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::emit_c()?)?;
        unsafe {
            OpaqueTypeRef::from_c_api(
                mlirEmitCOpaqueTypeGet(*self.handle.borrow(), StringRef::from(value.as_ref()).to_c_api()),
                self,
            )
            .map_err(|_| Error::invalid_argument("invalid arguments to `Context::emit_c_opaque_type`"))
        }
    }

    /// Creates a new Emit-C [`PointerTypeRef`] owned by this [`Context`].
    pub fn emit_c_pointer_type<'c, T: Type<'c, 't>>(&'c self, pointee: T) -> Result<PointerTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::emit_c()?)?;
        unsafe {
            PointerTypeRef::from_c_api(mlirEmitCPointerTypeGet(pointee.to_c_api()), self)
                .map_err(|_| Error::invalid_argument("invalid arguments to `Context::emit_c_pointer_type`"))
        }
    }

    /// Creates a new Emit-C [`SignedSizeTTypeRef`] owned by this [`Context`].
    pub fn emit_c_signed_size_t_type<'c>(&'c self) -> Result<SignedSizeTTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::emit_c()?)?;
        unsafe {
            SignedSizeTTypeRef::from_c_api(mlirEmitCSignedSizeTTypeGet(*self.handle.borrow()), self)
                .map_err(|_| Error::internal("invalid EmitC signed size type"))
        }
    }

    /// Creates a new Emit-C [`PtrDiffTTypeRef`] owned by this [`Context`].
    pub fn emit_c_ptrdiff_t_type<'c>(&'c self) -> Result<PtrDiffTTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::emit_c()?)?;
        unsafe {
            PtrDiffTTypeRef::from_c_api(mlirEmitCPtrDiffTTypeGet(*self.handle.borrow()), self)
                .map_err(|_| Error::internal("invalid EmitC pointer difference type"))
        }
    }

    /// Creates a new Emit-C [`SizeTTypeRef`] owned by this [`Context`].
    pub fn emit_c_size_t_type<'c>(&'c self) -> Result<SizeTTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::emit_c()?)?;
        unsafe {
            SizeTTypeRef::from_c_api(mlirEmitCSizeTTypeGet(*self.handle.borrow()), self)
                .map_err(|_| Error::internal("invalid EmitC size type"))
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_array_type() {
        let context = Context::new();
        let array_type = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]).unwrap();
        assert_eq!(&context, array_type.context());
        assert_eq!(array_type.dialect().unwrap().namespace().unwrap(), "emitc");
        assert_eq!(array_type.type_id().unwrap(), ArrayTypeRef::type_id().unwrap());
        assert_eq!(array_type.rank().unwrap(), 2);
        assert_eq!(array_type.shape().unwrap(), vec![4, 8]);
        assert_eq!(array_type.dimension(0).unwrap(), 4);
        assert_eq!(array_type.dimension(1).unwrap(), 8);
        assert!(array_type.dimension(2).is_err());
        assert_eq!(array_type.element_type().unwrap(), context.signless_integer_type(32));
    }

    #[test]
    fn test_array_type_equality() {
        let context = Context::new();
        let array_type_1 = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]).unwrap();
        let array_type_2 = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]).unwrap();
        assert_eq!(array_type_1, array_type_2);

        let array_type_2 = context.emit_c_array_type(context.signless_integer_type(32), &[8, 4]).unwrap();
        assert_ne!(array_type_1, array_type_2);

        let context = Context::new();
        let array_type_2 = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]).unwrap();
        assert_ne!(array_type_1, array_type_2);
    }

    #[test]
    fn test_array_type_display_and_debug() {
        let context = Context::new();
        let array_type = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]).unwrap();
        test_type_display_and_debug(array_type, "!emitc.array<4x8xi32>");
    }

    #[test]
    fn test_array_type_parsing() {
        let context = Context::new();
        let array_type = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]).unwrap();
        assert_eq!(context.parse_type("!emitc.array<4x8xi32>").unwrap(), array_type);
    }

    #[test]
    fn test_array_type_casting() {
        let context = Context::new();
        let array_type = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]).unwrap();
        test_type_casting(array_type);
    }

    #[test]
    fn test_lvalue_type() {
        let context = Context::new();
        let lvalue_type = context.emit_c_lvalue_type(context.signless_integer_type(32)).unwrap();
        assert_eq!(&context, lvalue_type.context());
        assert_eq!(lvalue_type.dialect().unwrap().namespace().unwrap(), "emitc");
        assert_eq!(lvalue_type.type_id().unwrap(), LValueTypeRef::type_id().unwrap());
    }

    #[test]
    fn test_lvalue_type_equality() {
        let context = Context::new();
        let lvalue_type_1 = context.emit_c_lvalue_type(context.signless_integer_type(32)).unwrap();
        let lvalue_type_2 = context.emit_c_lvalue_type(context.signless_integer_type(32)).unwrap();
        assert_eq!(lvalue_type_1, lvalue_type_2);

        let lvalue_type_2 = context.emit_c_lvalue_type(context.signless_integer_type(64)).unwrap();
        assert_ne!(lvalue_type_1, lvalue_type_2);

        let context = Context::new();
        let lvalue_type_2 = context.emit_c_lvalue_type(context.signless_integer_type(32)).unwrap();
        assert_ne!(lvalue_type_1, lvalue_type_2);
    }

    #[test]
    fn test_lvalue_type_display_and_debug() {
        let context = Context::new();
        let lvalue_type = context.emit_c_lvalue_type(context.signless_integer_type(32)).unwrap();
        test_type_display_and_debug(lvalue_type, "!emitc.lvalue<i32>");
    }

    #[test]
    fn test_lvalue_type_parsing() {
        let context = Context::new();
        let lvalue_type = context.emit_c_lvalue_type(context.signless_integer_type(32)).unwrap();
        assert_eq!(context.parse_type("!emitc.lvalue<i32>").unwrap(), lvalue_type);
    }

    #[test]
    fn test_lvalue_type_casting() {
        let context = Context::new();
        let lvalue_type = context.emit_c_lvalue_type(context.signless_integer_type(32)).unwrap();
        test_type_casting(lvalue_type);
    }

    #[test]
    fn test_opaque_type() {
        let context = Context::new();
        let opaque_type = context.emit_c_opaque_type("std::byte").unwrap();
        assert_eq!(&context, opaque_type.context());
        assert_eq!(opaque_type.dialect().unwrap().namespace().unwrap(), "emitc");
        assert_eq!(opaque_type.type_id().unwrap(), OpaqueTypeRef::type_id().unwrap());
    }

    #[test]
    fn test_opaque_type_equality() {
        let context = Context::new();
        let opaque_type_1 = context.emit_c_opaque_type("std::byte").unwrap();
        let opaque_type_2 = context.emit_c_opaque_type("std::byte").unwrap();
        assert_eq!(opaque_type_1, opaque_type_2);

        let opaque_type_2 = context.emit_c_opaque_type("std::int32_t").unwrap();
        assert_ne!(opaque_type_1, opaque_type_2);

        let context = Context::new();
        let opaque_type_2 = context.emit_c_opaque_type("std::byte").unwrap();
        assert_ne!(opaque_type_1, opaque_type_2);
    }

    #[test]
    fn test_opaque_type_display_and_debug() {
        let context = Context::new();
        let opaque_type = context.emit_c_opaque_type("std::byte").unwrap();
        test_type_display_and_debug(opaque_type, "!emitc.opaque<\"std::byte\">");
    }

    #[test]
    fn test_opaque_type_parsing() {
        let context = Context::new();
        let opaque_type = context.emit_c_opaque_type("std::byte").unwrap();
        assert_eq!(context.parse_type("!emitc.opaque<\"std::byte\">").unwrap(), opaque_type);
    }

    #[test]
    fn test_opaque_type_casting() {
        let context = Context::new();
        let opaque_type = context.emit_c_opaque_type("std::byte").unwrap();
        test_type_casting(opaque_type);
    }

    #[test]
    fn test_pointer_type() {
        let context = Context::new();
        let pointer_type = context.emit_c_pointer_type(context.signless_integer_type(32)).unwrap();
        assert_eq!(&context, pointer_type.context());
        assert_eq!(pointer_type.dialect().unwrap().namespace().unwrap(), "emitc");
        assert_eq!(pointer_type.type_id().unwrap(), PointerTypeRef::type_id().unwrap());
    }

    #[test]
    fn test_pointer_type_equality() {
        let context = Context::new();
        let pointer_type_1 = context.emit_c_pointer_type(context.signless_integer_type(32)).unwrap();
        let pointer_type_2 = context.emit_c_pointer_type(context.signless_integer_type(32)).unwrap();
        assert_eq!(pointer_type_1, pointer_type_2);

        let pointer_type_2 = context.emit_c_pointer_type(context.signless_integer_type(64)).unwrap();
        assert_ne!(pointer_type_1, pointer_type_2);

        let context = Context::new();
        let pointer_type_2 = context.emit_c_pointer_type(context.signless_integer_type(32)).unwrap();
        assert_ne!(pointer_type_1, pointer_type_2);
    }

    #[test]
    fn test_pointer_type_display_and_debug() {
        let context = Context::new();
        let pointer_type = context.emit_c_pointer_type(context.signless_integer_type(32)).unwrap();
        test_type_display_and_debug(pointer_type, "!emitc.ptr<i32>");
    }

    #[test]
    fn test_pointer_type_parsing() {
        let context = Context::new();
        let pointer_type = context.emit_c_pointer_type(context.signless_integer_type(32)).unwrap();
        assert_eq!(context.parse_type("!emitc.ptr<i32>").unwrap(), pointer_type);
    }

    #[test]
    fn test_pointer_type_casting() {
        let context = Context::new();
        let pointer_type = context.emit_c_pointer_type(context.signless_integer_type(32)).unwrap();
        test_type_casting(pointer_type);
    }

    #[test]
    fn test_signed_size_t_type() {
        let context = Context::new();
        let type_ref = context.emit_c_signed_size_t_type().unwrap();
        assert_eq!(&context, type_ref.context());
        assert_eq!(type_ref.dialect().unwrap().namespace().unwrap(), "emitc");
        assert_eq!(type_ref.type_id().unwrap(), SignedSizeTTypeRef::type_id().unwrap());
    }

    #[test]
    fn test_signed_size_t_type_equality() {
        let context = Context::new();
        let type_ref_1 = context.emit_c_signed_size_t_type().unwrap();
        let type_ref_2 = context.emit_c_signed_size_t_type().unwrap();
        assert_eq!(type_ref_1, type_ref_2);

        let context = Context::new();
        let type_ref_2 = context.emit_c_signed_size_t_type().unwrap();
        assert_ne!(type_ref_1, type_ref_2);
    }

    #[test]
    fn test_signed_size_t_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.emit_c_signed_size_t_type().unwrap(), "!emitc.ssize_t");
    }

    #[test]
    fn test_signed_size_t_type_parsing() {
        let context = Context::new();
        let type_ref = context.emit_c_signed_size_t_type().unwrap();
        assert_eq!(context.parse_type("!emitc.ssize_t").unwrap(), type_ref);
    }

    #[test]
    fn test_signed_size_t_type_casting() {
        let context = Context::new();
        test_type_casting(context.emit_c_signed_size_t_type().unwrap());
    }

    #[test]
    fn test_ptrdiff_t_type() {
        let context = Context::new();
        let type_ref = context.emit_c_ptrdiff_t_type().unwrap();
        assert_eq!(&context, type_ref.context());
        assert_eq!(type_ref.dialect().unwrap().namespace().unwrap(), "emitc");
        assert_eq!(type_ref.type_id().unwrap(), PtrDiffTTypeRef::type_id().unwrap());
    }

    #[test]
    fn test_ptrdiff_t_type_equality() {
        let context = Context::new();
        let type_ref_1 = context.emit_c_ptrdiff_t_type().unwrap();
        let type_ref_2 = context.emit_c_ptrdiff_t_type().unwrap();
        assert_eq!(type_ref_1, type_ref_2);

        let context = Context::new();
        let type_ref_2 = context.emit_c_ptrdiff_t_type().unwrap();
        assert_ne!(type_ref_1, type_ref_2);
    }

    #[test]
    fn test_ptrdiff_t_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.emit_c_ptrdiff_t_type().unwrap(), "!emitc.ptrdiff_t");
    }

    #[test]
    fn test_ptrdiff_t_type_parsing() {
        let context = Context::new();
        let type_ref = context.emit_c_ptrdiff_t_type().unwrap();
        assert_eq!(context.parse_type("!emitc.ptrdiff_t").unwrap(), type_ref);
    }

    #[test]
    fn test_ptrdiff_t_type_casting() {
        let context = Context::new();
        test_type_casting(context.emit_c_ptrdiff_t_type().unwrap());
    }

    #[test]
    fn test_size_t_type() {
        let context = Context::new();
        let type_ref = context.emit_c_size_t_type().unwrap();
        assert_eq!(&context, type_ref.context());
        assert_eq!(type_ref.dialect().unwrap().namespace().unwrap(), "emitc");
        assert_eq!(type_ref.type_id().unwrap(), SizeTTypeRef::type_id().unwrap());
    }

    #[test]
    fn test_size_t_type_equality() {
        let context = Context::new();
        let type_ref_1 = context.emit_c_size_t_type().unwrap();
        let type_ref_2 = context.emit_c_size_t_type().unwrap();
        assert_eq!(type_ref_1, type_ref_2);

        let context = Context::new();
        let type_ref_2 = context.emit_c_size_t_type().unwrap();
        assert_ne!(type_ref_1, type_ref_2);
    }

    #[test]
    fn test_size_t_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.emit_c_size_t_type().unwrap(), "!emitc.size_t");
    }

    #[test]
    fn test_size_t_type_parsing() {
        let context = Context::new();
        let type_ref = context.emit_c_size_t_type().unwrap();
        assert_eq!(context.parse_type("!emitc.size_t").unwrap(), type_ref);
    }

    #[test]
    fn test_size_t_type_casting() {
        let context = Context::new();
        test_type_casting(context.emit_c_size_t_type().unwrap());
    }
}
