use ryft_xla_sys::bindings::{
    MlirType, mlirEmitCArrayTypeGet, mlirEmitCArrayTypeGetTypeID, mlirEmitCLValueTypeGet, mlirEmitCLValueTypeGetTypeID,
    mlirEmitCOpaqueTypeGet, mlirEmitCOpaqueTypeGetTypeID, mlirEmitCPointerTypeGet, mlirEmitCPointerTypeGetTypeID,
    mlirEmitCPtrDiffTTypeGet, mlirEmitCPtrDiffTTypeGetTypeID, mlirEmitCSignedSizeTTypeGet,
    mlirEmitCSignedSizeTTypeGetTypeID, mlirEmitCSizeTTypeGet, mlirEmitCSizeTTypeGetTypeID, mlirShapedTypeGetDimSize,
    mlirShapedTypeGetElementType, mlirShapedTypeGetRank, mlirTypeIsAEmitCArrayType, mlirTypeIsAEmitCLValueType,
    mlirTypeIsAEmitCOpaqueType, mlirTypeIsAEmitCPointerType, mlirTypeIsAEmitCPtrDiffTType,
    mlirTypeIsAEmitCSignedSizeTType, mlirTypeIsAEmitCSizeTType,
};

use crate::{Context, DialectHandle, ShapedType, StringRef, Type, TypeId, TypeRef, mlir_subtype_trait_impls};

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
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirEmitCArrayTypeGetTypeID()).unwrap() }
    }

    /// Returns the rank of this array type.
    pub fn rank(&self) -> usize {
        usize::try_from(unsafe { mlirShapedTypeGetRank(self.handle) }).expect("invalid EmitC array rank")
    }

    /// Returns the shape of this array type.
    pub fn shape(&self) -> Vec<usize> {
        (0..self.rank()).map(|dimension| self.dimension(dimension)).collect()
    }

    /// Returns the `dimension`-th static dimension of this array type.
    pub fn dimension(&self, dimension: usize) -> usize {
        if dimension >= self.rank() {
            panic!("dimension is out of bounds");
        }
        usize::try_from(unsafe { mlirShapedTypeGetDimSize(self.handle, dimension.cast_signed()) })
            .expect("invalid EmitC array dimension")
    }

    /// Returns the element [`Type`] of this array type.
    pub fn element_type(&self) -> TypeRef<'_, '_> {
        unsafe { TypeRef::from_c_api(mlirShapedTypeGetElementType(self.handle), self.context).unwrap() }
    }
}

impl<'c, 't> Type<'c, 't> for ArrayTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        if unsafe { mlirTypeIsAEmitCArrayType(handle) } { Some(Self { handle, context }) } else { None }
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
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirEmitCLValueTypeGetTypeID()).unwrap() }
    }
}

impl<'c, 't> Type<'c, 't> for LValueTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        if unsafe { mlirTypeIsAEmitCLValueType(handle) } { Some(Self { handle, context }) } else { None }
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
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirEmitCOpaqueTypeGetTypeID()).unwrap() }
    }
}

impl<'c, 't> Type<'c, 't> for OpaqueTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        if unsafe { mlirTypeIsAEmitCOpaqueType(handle) } { Some(Self { handle, context }) } else { None }
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
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirEmitCPointerTypeGetTypeID()).unwrap() }
    }
}

impl<'c, 't> Type<'c, 't> for PointerTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        if unsafe { mlirTypeIsAEmitCPointerType(handle) } { Some(Self { handle, context }) } else { None }
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
            pub fn type_id() -> TypeId<'static> {
                unsafe { TypeId::from_c_api($get_type_id()).unwrap() }
            }
        }

        impl<'c, 't> Type<'c, 't> for $name<'c, 't> {
            unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
                if handle.ptr.is_null() {
                    return None;
                }
                if unsafe { $is_a(handle) } { Some(Self { handle, context }) } else { None }
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
    pub fn emit_c_array_type<'c, T: Type<'c, 't>>(&'c self, element_type: T, shape: &[usize]) -> ArrayTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::emit_c());
        let mut shape = shape.iter().map(|dimension| *dimension as i64).collect::<Vec<_>>();
        unsafe {
            ArrayTypeRef::from_c_api(
                mlirEmitCArrayTypeGet(shape.len().cast_signed(), shape.as_mut_ptr(), element_type.to_c_api()),
                self,
            )
            .expect("invalid EmitC array type")
        }
    }

    /// Creates a new Emit-C [`LValueTypeRef`] owned by this [`Context`].
    pub fn emit_c_lvalue_type<'c, T: Type<'c, 't>>(&'c self, value_type: T) -> LValueTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::emit_c());
        unsafe {
            LValueTypeRef::from_c_api(mlirEmitCLValueTypeGet(value_type.to_c_api()), self)
                .expect("invalid EmitC lvalue type")
        }
    }

    /// Creates a new Emit-C [`OpaqueTypeRef`] owned by this [`Context`].
    pub fn emit_c_opaque_type<'c, S: AsRef<str>>(&'c self, value: S) -> OpaqueTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::emit_c());
        unsafe {
            OpaqueTypeRef::from_c_api(
                mlirEmitCOpaqueTypeGet(*self.handle.borrow(), StringRef::from(value.as_ref()).to_c_api()),
                self,
            )
            .expect("invalid EmitC opaque type")
        }
    }

    /// Creates a new Emit-C [`PointerTypeRef`] owned by this [`Context`].
    pub fn emit_c_pointer_type<'c, T: Type<'c, 't>>(&'c self, pointee: T) -> PointerTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::emit_c());
        unsafe {
            PointerTypeRef::from_c_api(mlirEmitCPointerTypeGet(pointee.to_c_api()), self)
                .expect("invalid EmitC pointer type")
        }
    }

    /// Creates a new Emit-C [`SignedSizeTTypeRef`] owned by this [`Context`].
    pub fn emit_c_signed_size_t_type<'c>(&'c self) -> SignedSizeTTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::emit_c());
        unsafe {
            SignedSizeTTypeRef::from_c_api(mlirEmitCSignedSizeTTypeGet(*self.handle.borrow()), self)
                .expect("invalid EmitC signed size type")
        }
    }

    /// Creates a new Emit-C [`PtrDiffTTypeRef`] owned by this [`Context`].
    pub fn emit_c_ptrdiff_t_type<'c>(&'c self) -> PtrDiffTTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::emit_c());
        unsafe {
            PtrDiffTTypeRef::from_c_api(mlirEmitCPtrDiffTTypeGet(*self.handle.borrow()), self)
                .expect("invalid EmitC pointer difference type")
        }
    }

    /// Creates a new Emit-C [`SizeTTypeRef`] owned by this [`Context`].
    pub fn emit_c_size_t_type<'c>(&'c self) -> SizeTTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::emit_c());
        unsafe {
            SizeTTypeRef::from_c_api(mlirEmitCSizeTTypeGet(*self.handle.borrow()), self)
                .expect("invalid EmitC size type")
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
        let array_type = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]);
        assert_eq!(&context, array_type.context());
        assert_eq!(array_type.dialect().namespace().unwrap(), "emitc");
        assert_eq!(array_type.type_id(), ArrayTypeRef::type_id());
        assert_eq!(array_type.rank(), 2);
        assert_eq!(array_type.shape(), vec![4, 8]);
        assert_eq!(array_type.dimension(0), 4);
        assert_eq!(array_type.dimension(1), 8);
        assert_eq!(array_type.element_type(), context.signless_integer_type(32));
    }

    #[test]
    fn test_array_type_equality() {
        let context = Context::new();
        let array_type_1 = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]);
        let array_type_2 = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]);
        assert_eq!(array_type_1, array_type_2);

        let array_type_2 = context.emit_c_array_type(context.signless_integer_type(32), &[8, 4]);
        assert_ne!(array_type_1, array_type_2);

        let context = Context::new();
        let array_type_2 = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]);
        assert_ne!(array_type_1, array_type_2);
    }

    #[test]
    fn test_array_type_display_and_debug() {
        let context = Context::new();
        let array_type = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]);
        test_type_display_and_debug(array_type, "!emitc.array<4x8xi32>");
    }

    #[test]
    fn test_array_type_parsing() {
        let context = Context::new();
        let array_type = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]);
        assert_eq!(context.parse_type("!emitc.array<4x8xi32>").unwrap(), array_type);
    }

    #[test]
    fn test_array_type_casting() {
        let context = Context::new();
        let array_type = context.emit_c_array_type(context.signless_integer_type(32), &[4, 8]);
        test_type_casting(array_type);
    }

    #[test]
    fn test_lvalue_type() {
        let context = Context::new();
        let lvalue_type = context.emit_c_lvalue_type(context.signless_integer_type(32));
        assert_eq!(&context, lvalue_type.context());
        assert_eq!(lvalue_type.dialect().namespace().unwrap(), "emitc");
        assert_eq!(lvalue_type.type_id(), LValueTypeRef::type_id());
    }

    #[test]
    fn test_lvalue_type_equality() {
        let context = Context::new();
        let lvalue_type_1 = context.emit_c_lvalue_type(context.signless_integer_type(32));
        let lvalue_type_2 = context.emit_c_lvalue_type(context.signless_integer_type(32));
        assert_eq!(lvalue_type_1, lvalue_type_2);

        let lvalue_type_2 = context.emit_c_lvalue_type(context.signless_integer_type(64));
        assert_ne!(lvalue_type_1, lvalue_type_2);

        let context = Context::new();
        let lvalue_type_2 = context.emit_c_lvalue_type(context.signless_integer_type(32));
        assert_ne!(lvalue_type_1, lvalue_type_2);
    }

    #[test]
    fn test_lvalue_type_display_and_debug() {
        let context = Context::new();
        let lvalue_type = context.emit_c_lvalue_type(context.signless_integer_type(32));
        test_type_display_and_debug(lvalue_type, "!emitc.lvalue<i32>");
    }

    #[test]
    fn test_lvalue_type_parsing() {
        let context = Context::new();
        let lvalue_type = context.emit_c_lvalue_type(context.signless_integer_type(32));
        assert_eq!(context.parse_type("!emitc.lvalue<i32>").unwrap(), lvalue_type);
    }

    #[test]
    fn test_lvalue_type_casting() {
        let context = Context::new();
        let lvalue_type = context.emit_c_lvalue_type(context.signless_integer_type(32));
        test_type_casting(lvalue_type);
    }

    #[test]
    fn test_opaque_type() {
        let context = Context::new();
        let opaque_type = context.emit_c_opaque_type("std::byte");
        assert_eq!(&context, opaque_type.context());
        assert_eq!(opaque_type.dialect().namespace().unwrap(), "emitc");
        assert_eq!(opaque_type.type_id(), OpaqueTypeRef::type_id());
    }

    #[test]
    fn test_opaque_type_equality() {
        let context = Context::new();
        let opaque_type_1 = context.emit_c_opaque_type("std::byte");
        let opaque_type_2 = context.emit_c_opaque_type("std::byte");
        assert_eq!(opaque_type_1, opaque_type_2);

        let opaque_type_2 = context.emit_c_opaque_type("std::int32_t");
        assert_ne!(opaque_type_1, opaque_type_2);

        let context = Context::new();
        let opaque_type_2 = context.emit_c_opaque_type("std::byte");
        assert_ne!(opaque_type_1, opaque_type_2);
    }

    #[test]
    fn test_opaque_type_display_and_debug() {
        let context = Context::new();
        let opaque_type = context.emit_c_opaque_type("std::byte");
        test_type_display_and_debug(opaque_type, "!emitc.opaque<\"std::byte\">");
    }

    #[test]
    fn test_opaque_type_parsing() {
        let context = Context::new();
        let opaque_type = context.emit_c_opaque_type("std::byte");
        assert_eq!(context.parse_type("!emitc.opaque<\"std::byte\">").unwrap(), opaque_type);
    }

    #[test]
    fn test_opaque_type_casting() {
        let context = Context::new();
        let opaque_type = context.emit_c_opaque_type("std::byte");
        test_type_casting(opaque_type);
    }

    #[test]
    fn test_pointer_type() {
        let context = Context::new();
        let pointer_type = context.emit_c_pointer_type(context.signless_integer_type(32));
        assert_eq!(&context, pointer_type.context());
        assert_eq!(pointer_type.dialect().namespace().unwrap(), "emitc");
        assert_eq!(pointer_type.type_id(), PointerTypeRef::type_id());
    }

    #[test]
    fn test_pointer_type_equality() {
        let context = Context::new();
        let pointer_type_1 = context.emit_c_pointer_type(context.signless_integer_type(32));
        let pointer_type_2 = context.emit_c_pointer_type(context.signless_integer_type(32));
        assert_eq!(pointer_type_1, pointer_type_2);

        let pointer_type_2 = context.emit_c_pointer_type(context.signless_integer_type(64));
        assert_ne!(pointer_type_1, pointer_type_2);

        let context = Context::new();
        let pointer_type_2 = context.emit_c_pointer_type(context.signless_integer_type(32));
        assert_ne!(pointer_type_1, pointer_type_2);
    }

    #[test]
    fn test_pointer_type_display_and_debug() {
        let context = Context::new();
        let pointer_type = context.emit_c_pointer_type(context.signless_integer_type(32));
        test_type_display_and_debug(pointer_type, "!emitc.ptr<i32>");
    }

    #[test]
    fn test_pointer_type_parsing() {
        let context = Context::new();
        let pointer_type = context.emit_c_pointer_type(context.signless_integer_type(32));
        assert_eq!(context.parse_type("!emitc.ptr<i32>").unwrap(), pointer_type);
    }

    #[test]
    fn test_pointer_type_casting() {
        let context = Context::new();
        let pointer_type = context.emit_c_pointer_type(context.signless_integer_type(32));
        test_type_casting(pointer_type);
    }

    #[test]
    fn test_signed_size_t_type() {
        let context = Context::new();
        let type_ref = context.emit_c_signed_size_t_type();
        assert_eq!(&context, type_ref.context());
        assert_eq!(type_ref.dialect().namespace().unwrap(), "emitc");
        assert_eq!(type_ref.type_id(), SignedSizeTTypeRef::type_id());
    }

    #[test]
    fn test_signed_size_t_type_equality() {
        let context = Context::new();
        let type_ref_1 = context.emit_c_signed_size_t_type();
        let type_ref_2 = context.emit_c_signed_size_t_type();
        assert_eq!(type_ref_1, type_ref_2);

        let context = Context::new();
        let type_ref_2 = context.emit_c_signed_size_t_type();
        assert_ne!(type_ref_1, type_ref_2);
    }

    #[test]
    fn test_signed_size_t_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.emit_c_signed_size_t_type(), "!emitc.ssize_t");
    }

    #[test]
    fn test_signed_size_t_type_parsing() {
        let context = Context::new();
        let type_ref = context.emit_c_signed_size_t_type();
        assert_eq!(context.parse_type("!emitc.ssize_t").unwrap(), type_ref);
    }

    #[test]
    fn test_signed_size_t_type_casting() {
        let context = Context::new();
        test_type_casting(context.emit_c_signed_size_t_type());
    }

    #[test]
    fn test_ptrdiff_t_type() {
        let context = Context::new();
        let type_ref = context.emit_c_ptrdiff_t_type();
        assert_eq!(&context, type_ref.context());
        assert_eq!(type_ref.dialect().namespace().unwrap(), "emitc");
        assert_eq!(type_ref.type_id(), PtrDiffTTypeRef::type_id());
    }

    #[test]
    fn test_ptrdiff_t_type_equality() {
        let context = Context::new();
        let type_ref_1 = context.emit_c_ptrdiff_t_type();
        let type_ref_2 = context.emit_c_ptrdiff_t_type();
        assert_eq!(type_ref_1, type_ref_2);

        let context = Context::new();
        let type_ref_2 = context.emit_c_ptrdiff_t_type();
        assert_ne!(type_ref_1, type_ref_2);
    }

    #[test]
    fn test_ptrdiff_t_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.emit_c_ptrdiff_t_type(), "!emitc.ptrdiff_t");
    }

    #[test]
    fn test_ptrdiff_t_type_parsing() {
        let context = Context::new();
        let type_ref = context.emit_c_ptrdiff_t_type();
        assert_eq!(context.parse_type("!emitc.ptrdiff_t").unwrap(), type_ref);
    }

    #[test]
    fn test_ptrdiff_t_type_casting() {
        let context = Context::new();
        test_type_casting(context.emit_c_ptrdiff_t_type());
    }

    #[test]
    fn test_size_t_type() {
        let context = Context::new();
        let type_ref = context.emit_c_size_t_type();
        assert_eq!(&context, type_ref.context());
        assert_eq!(type_ref.dialect().namespace().unwrap(), "emitc");
        assert_eq!(type_ref.type_id(), SizeTTypeRef::type_id());
    }

    #[test]
    fn test_size_t_type_equality() {
        let context = Context::new();
        let type_ref_1 = context.emit_c_size_t_type();
        let type_ref_2 = context.emit_c_size_t_type();
        assert_eq!(type_ref_1, type_ref_2);

        let context = Context::new();
        let type_ref_2 = context.emit_c_size_t_type();
        assert_ne!(type_ref_1, type_ref_2);
    }

    #[test]
    fn test_size_t_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.emit_c_size_t_type(), "!emitc.size_t");
    }

    #[test]
    fn test_size_t_type_parsing() {
        let context = Context::new();
        let type_ref = context.emit_c_size_t_type();
        assert_eq!(context.parse_type("!emitc.size_t").unwrap(), type_ref);
    }

    #[test]
    fn test_size_t_type_casting() {
        let context = Context::new();
        test_type_casting(context.emit_c_size_t_type());
    }
}
