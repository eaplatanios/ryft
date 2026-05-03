use ryft_xla_sys::bindings::{
    MlirType, mlirLLVMArrayTypeGet, mlirLLVMArrayTypeGetElementType, mlirLLVMFunctionTypeGet,
    mlirLLVMFunctionTypeGetInput, mlirLLVMFunctionTypeGetNumInputs, mlirLLVMFunctionTypeGetReturnType,
    mlirLLVMPointerTypeGet, mlirLLVMPointerTypeGetAddressSpace, mlirLLVMPointerTypeGetTypeID,
    mlirLLVMStructTypeGetElementType, mlirLLVMStructTypeGetIdentifier, mlirLLVMStructTypeGetNumElementTypes,
    mlirLLVMStructTypeGetTypeID, mlirLLVMStructTypeIdentifiedGet, mlirLLVMStructTypeIdentifiedNewGet,
    mlirLLVMStructTypeIsLiteral, mlirLLVMStructTypeIsOpaque, mlirLLVMStructTypeIsPacked, mlirLLVMStructTypeLiteralGet,
    mlirLLVMStructTypeOpaqueGet, mlirLLVMStructTypeSetBody, mlirLLVMVoidTypeGet, mlirTypeIsALLVMPointerType,
    mlirTypeIsALLVMStructType,
};
use ryft_xla_sys::mlir::dialects::llvm::{
    mlirLLVMArrayTypeGetNumElements, mlirLLVMArrayTypeGetTypeID, mlirLLVMFunctionTypeGetTypeID,
    mlirLLVMFunctionTypeIsVarArg, mlirLlvmLabelTypeGet, mlirLlvmMetadataTypeGet, mlirLlvmPpcFp128TypeGet,
    mlirLlvmTargetExtTypeGet, mlirLlvmTargetExtTypeGetIntParam, mlirLlvmTargetExtTypeGetName,
    mlirLlvmTargetExtTypeGetNumIntParams, mlirLlvmTargetExtTypeGetNumTypeParams, mlirLlvmTargetExtTypeGetTypeParam,
    mlirLlvmTokenTypeGet, mlirLlvmX86AmxTypeGet, mlirTypeIsALLVMArrayType, mlirTypeIsALLVMFunctionType,
    mlirTypeIsALlvmLabelType, mlirTypeIsALlvmMetadataType, mlirTypeIsALlvmPpcFp128Type, mlirTypeIsALlvmTargetExtType,
    mlirTypeIsALlvmTokenType, mlirTypeIsALlvmVoidType, mlirTypeIsALlvmX86AmxType,
};

use crate::{Context, DialectHandle, LogicalResult, StringRef, Type, TypeId, TypeRef, mlir_subtype_trait_impls};

macro_rules! llvm_trivial_type {
    ($type_name:ident, $context_method:ident, $is_type:path, $get:path, $description:literal $(,)*) => {
        #[doc = "LLVM "]
        #[doc = $description]
        #[doc = " [`Type`]."]
        #[derive(Copy, Clone)]
        pub struct $type_name<'c, 't> {
            /// Handle that represents this [`Type`] in the MLIR C API.
            handle: MlirType,

            /// [`Context`] that owns this [`Type`].
            context: &'c Context<'t>,
        }

        impl<'c, 't> Type<'c, 't> for $type_name<'c, 't> {
            unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
                if handle.ptr.is_null() {
                    return None;
                }
                if unsafe { $is_type(handle) } {
                    Some(Self { handle, context })
                } else {
                    None
                }
            }

            unsafe fn to_c_api(&self) -> MlirType {
                self.handle
            }

            fn context(&self) -> &'c Context<'t> {
                self.context
            }
        }

        mlir_subtype_trait_impls!($type_name<'c, 't> as Type, mlir_type = Type);

        impl<'t> Context<'t> {
            #[doc = "Creates a new LLVM "]
            #[doc = $description]
            #[doc = " type owned by this [`Context`]."]
            pub fn $context_method<'c>(&'c self) -> $type_name<'c, 't> {
                self.load_dialect(DialectHandle::llvm());
                unsafe {
                    $type_name::from_c_api($get(*self.handle.borrow()), self)
                        .expect(concat!("invalid LLVM ", $description, " type"))
                }
            }
        }
    };
}

/// LLVM pointer [`Type`].
///
/// `!llvm.ptr` represents an opaque pointer value. Pointers optionally carry an integer address space.
///
/// Refer to the [official MLIR LLVM dialect documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmpointertype)
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
        unsafe { TypeId::from_c_api(mlirLLVMPointerTypeGetTypeID()).unwrap() }
    }

    /// Returns the address space of this pointer type.
    pub fn address_space(&self) -> u32 {
        unsafe { mlirLLVMPointerTypeGetAddressSpace(self.handle) }
    }
}

impl<'c, 't> Type<'c, 't> for PointerTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        if unsafe { mlirTypeIsALLVMPointerType(handle) } { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(PointerTypeRef<'c, 't> as Type, mlir_type = Type);

llvm_trivial_type!(VoidTypeRef, llvm_void_type, mlirTypeIsALlvmVoidType, mlirLLVMVoidTypeGet, "void");
llvm_trivial_type!(TokenTypeRef, llvm_token_type, mlirTypeIsALlvmTokenType, mlirLlvmTokenTypeGet, "token");
llvm_trivial_type!(LabelTypeRef, llvm_label_type, mlirTypeIsALlvmLabelType, mlirLlvmLabelTypeGet, "label");
llvm_trivial_type!(
    MetadataTypeRef,
    llvm_metadata_type,
    mlirTypeIsALlvmMetadataType,
    mlirLlvmMetadataTypeGet,
    "metadata",
);

/// LLVM array [`Type`].
///
/// `!llvm.array` represents a fixed-size aggregate containing elements of one LLVM-compatible type.
///
/// Refer to the [official MLIR LLVM dialect documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmarraytype)
/// for more information.
#[derive(Copy, Clone)]
pub struct ArrayTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> ArrayTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`ArrayTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirLLVMArrayTypeGetTypeID()).unwrap() }
    }

    /// Returns the number of elements in this array type.
    pub fn element_count(&self) -> u64 {
        u64::from(unsafe { mlirLLVMArrayTypeGetNumElements(self.handle) })
    }

    /// Returns the element [`Type`] of this array type.
    pub fn element_type(&self) -> TypeRef<'c, 't> {
        unsafe { TypeRef::from_c_api(mlirLLVMArrayTypeGetElementType(self.handle), self.context).unwrap() }
    }
}

impl<'c, 't> Type<'c, 't> for ArrayTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        if unsafe { mlirTypeIsALLVMArrayType(handle) } { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(ArrayTypeRef<'c, 't> as Type, mlir_type = Type);

/// LLVM function [`Type`].
///
/// `!llvm.func` stores one return type, zero or more input types, and optional variadic-call information.
///
/// Refer to the [official MLIR LLVM dialect documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfunctiontype)
/// for more information.
#[derive(Copy, Clone)]
pub struct FunctionTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> FunctionTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`FunctionTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirLLVMFunctionTypeGetTypeID()).unwrap() }
    }

    /// Returns the number of input parameters of this function type.
    pub fn input_count(&self) -> usize {
        usize::try_from(unsafe { mlirLLVMFunctionTypeGetNumInputs(self.handle) })
            .expect("invalid `!llvm.func` input count")
    }

    /// Returns `true` if this function type accepts a variadic argument tail.
    pub fn is_variadic(&self) -> bool {
        unsafe { mlirLLVMFunctionTypeIsVarArg(self.handle) }
    }

    /// Returns the input parameter [`Type`]s of this function type.
    pub fn inputs(&self) -> impl Iterator<Item = TypeRef<'c, 't>> {
        (0..self.input_count()).map(|index| self.input(index))
    }

    /// Returns the `index`-th input parameter [`Type`] of this function type.
    pub fn input(&self, index: usize) -> TypeRef<'c, 't> {
        if index >= self.input_count() {
            panic!("LLVM function type input index is out of bounds");
        }
        unsafe {
            TypeRef::from_c_api(mlirLLVMFunctionTypeGetInput(self.handle, index.cast_signed()), self.context).unwrap()
        }
    }

    /// Returns the return [`Type`] of this function type.
    pub fn return_type(&self) -> TypeRef<'c, 't> {
        unsafe { TypeRef::from_c_api(mlirLLVMFunctionTypeGetReturnType(self.handle), self.context).unwrap() }
    }
}

impl<'c, 't> Type<'c, 't> for FunctionTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        if unsafe { mlirTypeIsALLVMFunctionType(handle) } { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(FunctionTypeRef<'c, 't> as Type, mlir_type = Type);

/// LLVM struct [`Type`].
///
/// `!llvm.struct` represents either a literal aggregate or an identified aggregate that can be opaque or initialized
/// with a body.
///
/// Refer to the [official MLIR LLVM dialect documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmstructtype)
/// for more information.
#[derive(Copy, Clone)]
pub struct StructTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> StructTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`StructTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirLLVMStructTypeGetTypeID()).unwrap() }
    }

    /// Returns `true` if this struct type is literal rather than identified.
    pub fn is_literal(&self) -> bool {
        unsafe { mlirLLVMStructTypeIsLiteral(self.handle) }
    }

    /// Returns `true` if this struct type is identified rather than literal.
    pub fn is_identified(&self) -> bool {
        !self.is_literal()
    }

    /// Returns `true` if this struct type is packed.
    pub fn is_packed(&self) -> bool {
        unsafe { mlirLLVMStructTypeIsPacked(self.handle) }
    }

    /// Returns `true` if this struct type is opaque or does not have a body yet.
    pub fn is_opaque(&self) -> bool {
        unsafe { mlirLLVMStructTypeIsOpaque(self.handle) }
    }

    /// Returns the identifier of this struct type if it is identified.
    pub fn identifier(&self) -> Option<StringRef<'c>> {
        if self.is_literal() {
            None
        } else {
            Some(unsafe { StringRef::from_c_api(mlirLLVMStructTypeGetIdentifier(self.handle)) })
        }
    }

    /// Returns the number of body elements in this struct type, or [`None`] if the struct is opaque.
    pub fn element_count(&self) -> Option<usize> {
        if self.is_opaque() {
            None
        } else {
            Some(
                usize::try_from(unsafe { mlirLLVMStructTypeGetNumElementTypes(self.handle) })
                    .expect("invalid `!llvm.struct` element count"),
            )
        }
    }

    /// Returns the body element [`Type`]s of this struct type.
    pub fn element_types(&self) -> Option<impl Iterator<Item = TypeRef<'c, 't>>> {
        self.element_count().map(|element_count| (0..element_count).map(|index| self.element_type(index)))
    }

    /// Returns the `index`-th body element [`Type`] of this struct type.
    pub fn element_type(&self, index: usize) -> TypeRef<'c, 't> {
        let element_count = self.element_count().expect("opaque `!llvm.struct` does not have body elements");
        if index >= element_count {
            panic!("LLVM struct type element index is out of bounds");
        }
        unsafe {
            TypeRef::from_c_api(mlirLLVMStructTypeGetElementType(self.handle, index.cast_signed()), self.context)
                .unwrap()
        }
    }

    /// Sets the body of this identified struct type if MLIR allows it.
    pub fn set_body<T: Type<'c, 't>>(&self, element_types: &[T], is_packed: bool) -> LogicalResult {
        let element_types =
            element_types.iter().map(|element_type| unsafe { element_type.to_c_api() }).collect::<Vec<_>>();
        unsafe {
            LogicalResult::from_c_api(mlirLLVMStructTypeSetBody(
                self.handle,
                element_types.len().cast_signed(),
                element_types.as_ptr(),
                is_packed,
            ))
        }
    }
}

impl<'c, 't> Type<'c, 't> for StructTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        if unsafe { mlirTypeIsALLVMStructType(handle) } { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(StructTypeRef<'c, 't> as Type, mlir_type = Type);

/// LLVM target extension [`Type`].
///
/// `!llvm.target` represents target-specific values with a string name and optional type/integer parameters.
///
/// Refer to the [official MLIR LLVM dialect documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmtargetexttype)
/// for more information.
#[derive(Copy, Clone)]
pub struct TargetExtTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> TargetExtTypeRef<'c, 't> {
    /// Returns the target extension type name.
    pub fn name(&self) -> StringRef<'c> {
        unsafe { StringRef::from_c_api(mlirLlvmTargetExtTypeGetName(self.handle)) }
    }

    /// Returns the number of type parameters.
    pub fn type_parameter_count(&self) -> usize {
        usize::try_from(unsafe { mlirLlvmTargetExtTypeGetNumTypeParams(self.handle) })
            .expect("invalid `!llvm.target` type parameter count")
    }

    /// Returns the type parameters.
    pub fn type_parameters(&self) -> impl Iterator<Item = TypeRef<'c, 't>> {
        (0..self.type_parameter_count()).map(|index| self.type_parameter(index))
    }

    /// Returns the `index`-th type parameter.
    pub fn type_parameter(&self, index: usize) -> TypeRef<'c, 't> {
        if index >= self.type_parameter_count() {
            panic!("LLVM target extension type parameter index is out of bounds");
        }
        unsafe {
            TypeRef::from_c_api(mlirLlvmTargetExtTypeGetTypeParam(self.handle, index.cast_signed()), self.context)
                .expect("invalid `!llvm.target` type parameter")
        }
    }

    /// Returns the number of integer parameters.
    pub fn integer_parameter_count(&self) -> usize {
        usize::try_from(unsafe { mlirLlvmTargetExtTypeGetNumIntParams(self.handle) })
            .expect("invalid `!llvm.target` integer parameter count")
    }

    /// Returns the integer parameters.
    pub fn integer_parameters(&self) -> impl Iterator<Item = u32> {
        (0..self.integer_parameter_count()).map(|index| self.integer_parameter(index))
    }

    /// Returns the `index`-th integer parameter.
    pub fn integer_parameter(&self, index: usize) -> u32 {
        if index >= self.integer_parameter_count() {
            panic!("LLVM target extension integer parameter index is out of bounds");
        }
        unsafe { mlirLlvmTargetExtTypeGetIntParam(self.handle, index.cast_signed()) }
    }
}

impl<'c, 't> Type<'c, 't> for TargetExtTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        if unsafe { mlirTypeIsALlvmTargetExtType(handle) } { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(TargetExtTypeRef<'c, 't> as Type, mlir_type = Type);

llvm_trivial_type!(X86AmxTypeRef, llvm_x86_amx_type, mlirTypeIsALlvmX86AmxType, mlirLlvmX86AmxTypeGet, "x86 AMX");
llvm_trivial_type!(
    PpcFp128TypeRef,
    llvm_ppc_fp128_type,
    mlirTypeIsALlvmPpcFp128Type,
    mlirLlvmPpcFp128TypeGet,
    "PowerPC 128-bit floating-point",
);

impl<'t> Context<'t> {
    /// Creates a new LLVM [`PointerTypeRef`] owned by this [`Context`].
    pub fn llvm_pointer_type<'c>(&'c self, address_space: u32) -> PointerTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        unsafe {
            PointerTypeRef::from_c_api(mlirLLVMPointerTypeGet(*self.handle.borrow(), address_space), self)
                .expect("invalid LLVM pointer type")
        }
    }

    /// Creates a new LLVM [`ArrayTypeRef`] owned by this [`Context`].
    pub fn llvm_array_type<'c, T: Type<'c, 't>>(&'c self, element_type: T, element_count: u64) -> ArrayTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        unsafe {
            ArrayTypeRef::from_c_api(
                mlirLLVMArrayTypeGet(
                    element_type.to_c_api(),
                    u32::try_from(element_count).expect("invalid LLVM array element count"),
                ),
                self,
            )
            .expect("invalid LLVM array type")
        }
    }

    /// Creates a new LLVM [`FunctionTypeRef`] owned by this [`Context`].
    pub fn llvm_function_type<'c, R: Type<'c, 't>, A: Type<'c, 't>>(
        &'c self,
        return_type: R,
        input_types: &[A],
        is_variadic: bool,
    ) -> FunctionTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        let input_types = input_types.iter().map(|input_type| unsafe { input_type.to_c_api() }).collect::<Vec<_>>();
        unsafe {
            FunctionTypeRef::from_c_api(
                mlirLLVMFunctionTypeGet(
                    return_type.to_c_api(),
                    input_types.len().cast_signed(),
                    input_types.as_ptr(),
                    is_variadic,
                ),
                self,
            )
            .expect("invalid LLVM function type")
        }
    }

    /// Creates a new literal LLVM [`StructTypeRef`] owned by this [`Context`].
    pub fn llvm_literal_struct_type<'c, T: Type<'c, 't>>(
        &'c self,
        element_types: &[T],
        is_packed: bool,
    ) -> StructTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        let element_types =
            element_types.iter().map(|element_type| unsafe { element_type.to_c_api() }).collect::<Vec<_>>();
        unsafe {
            StructTypeRef::from_c_api(
                mlirLLVMStructTypeLiteralGet(
                    *self.handle.borrow(),
                    element_types.len().cast_signed(),
                    element_types.as_ptr(),
                    is_packed,
                ),
                self,
            )
            .expect("invalid LLVM literal struct type")
        }
    }

    /// Creates or retrieves an identified LLVM [`StructTypeRef`] with no body.
    pub fn llvm_identified_struct_type<'c, S: AsRef<str>>(&'c self, name: S) -> StructTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        unsafe {
            StructTypeRef::from_c_api(
                mlirLLVMStructTypeIdentifiedGet(*self.handle.borrow(), StringRef::from(name.as_ref()).to_c_api()),
                self,
            )
            .expect("invalid LLVM identified struct type")
        }
    }

    /// Creates a fresh identified LLVM [`StructTypeRef`] with a body.
    pub fn llvm_new_identified_struct_type<'c, S: AsRef<str>, T: Type<'c, 't>>(
        &'c self,
        name: S,
        element_types: &[T],
        is_packed: bool,
    ) -> StructTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        let element_types =
            element_types.iter().map(|element_type| unsafe { element_type.to_c_api() }).collect::<Vec<_>>();
        unsafe {
            StructTypeRef::from_c_api(
                mlirLLVMStructTypeIdentifiedNewGet(
                    *self.handle.borrow(),
                    StringRef::from(name.as_ref()).to_c_api(),
                    element_types.len().cast_signed(),
                    element_types.as_ptr(),
                    is_packed,
                ),
                self,
            )
            .expect("invalid LLVM new identified struct type")
        }
    }

    /// Creates or retrieves an intentionally opaque identified LLVM [`StructTypeRef`].
    pub fn llvm_opaque_struct_type<'c, S: AsRef<str>>(&'c self, name: S) -> StructTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        unsafe {
            StructTypeRef::from_c_api(
                mlirLLVMStructTypeOpaqueGet(*self.handle.borrow(), StringRef::from(name.as_ref()).to_c_api()),
                self,
            )
            .expect("invalid LLVM opaque struct type")
        }
    }

    /// Creates a new LLVM [`TargetExtTypeRef`] owned by this [`Context`].
    ///
    /// # Parameters
    ///
    ///   - `name`: Target extension type name.
    ///   - `type_parameters`: Type parameters whose interpretation is target-defined.
    ///   - `integer_parameters`: Integer parameters whose interpretation is target-defined.
    pub fn llvm_target_ext_type<'c, S: AsRef<str>>(
        &'c self,
        name: S,
        type_parameters: &[TypeRef<'c, 't>],
        integer_parameters: &[u32],
    ) -> TargetExtTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        let name = StringRef::from(name.as_ref());
        let type_parameters = type_parameters
            .iter()
            .map(|type_parameter| unsafe { type_parameter.to_c_api() })
            .collect::<Vec<_>>();
        unsafe {
            TargetExtTypeRef::from_c_api(
                mlirLlvmTargetExtTypeGet(
                    *self.handle.borrow(),
                    name.to_c_api(),
                    type_parameters.len().cast_signed(),
                    type_parameters.as_ptr(),
                    integer_parameters.len().cast_signed(),
                    integer_parameters.as_ptr(),
                ),
                self,
            )
            .expect("invalid LLVM target extension type")
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_pointer_type() {
        let context = Context::new();
        let pointer_type = context.llvm_pointer_type(3);
        assert_eq!(&context, pointer_type.context());
        assert_eq!(pointer_type.dialect().namespace().unwrap(), "llvm");
        assert_eq!(pointer_type.type_id(), PointerTypeRef::type_id());
        assert_eq!(pointer_type.address_space(), 3);
    }

    #[test]
    fn test_pointer_type_equality() {
        let context = Context::new();
        let pointer_type_1 = context.llvm_pointer_type(0);
        let pointer_type_2 = context.llvm_pointer_type(0);
        assert_eq!(pointer_type_1, pointer_type_2);

        let pointer_type_2 = context.llvm_pointer_type(1);
        assert_ne!(pointer_type_1, pointer_type_2);

        let context = Context::new();
        let pointer_type_2 = context.llvm_pointer_type(0);
        assert_ne!(pointer_type_1, pointer_type_2);
    }

    #[test]
    fn test_pointer_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.llvm_pointer_type(0), "!llvm.ptr");
        test_type_display_and_debug(context.llvm_pointer_type(3), "!llvm.ptr<3>");
    }

    #[test]
    fn test_pointer_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        assert_eq!(context.parse_type("!llvm.ptr").unwrap(), context.llvm_pointer_type(0));
        assert_eq!(context.parse_type("!llvm.ptr<3>").unwrap(), context.llvm_pointer_type(3));
    }

    #[test]
    fn test_pointer_type_casting() {
        let context = Context::new();
        test_type_casting(context.llvm_pointer_type(0));
    }

    #[test]
    fn test_void_type() {
        let context = Context::new();
        let void_type = context.llvm_void_type();
        assert_eq!(&context, void_type.context());
        assert_eq!(void_type.dialect().namespace().unwrap(), "llvm");
    }

    #[test]
    fn test_void_type_equality() {
        let context = Context::new();
        let void_type_1 = context.llvm_void_type();
        let void_type_2 = context.llvm_void_type();
        assert_eq!(void_type_1, void_type_2);

        let context = Context::new();
        let void_type_2 = context.llvm_void_type();
        assert_ne!(void_type_1, void_type_2);
    }

    #[test]
    fn test_void_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.llvm_void_type(), "!llvm.void");
    }

    #[test]
    fn test_void_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        assert_eq!(context.parse_type("!llvm.void").unwrap(), context.llvm_void_type());
    }

    #[test]
    fn test_void_type_casting() {
        let context = Context::new();
        test_type_casting(context.llvm_void_type());
    }

    #[test]
    fn test_token_type() {
        let context = Context::new();
        let token_type = context.llvm_token_type();
        assert_eq!(&context, token_type.context());
        assert_eq!(token_type.dialect().namespace().unwrap(), "llvm");
    }

    #[test]
    fn test_token_type_equality() {
        let context = Context::new();
        let token_type_1 = context.llvm_token_type();
        let token_type_2 = context.llvm_token_type();
        assert_eq!(token_type_1, token_type_2);

        let context = Context::new();
        let token_type_2 = context.llvm_token_type();
        assert_ne!(token_type_1, token_type_2);
    }

    #[test]
    fn test_token_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.llvm_token_type(), "!llvm.token");
    }

    #[test]
    fn test_token_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        assert_eq!(context.parse_type("!llvm.token").unwrap(), context.llvm_token_type());
    }

    #[test]
    fn test_token_type_casting() {
        let context = Context::new();
        test_type_casting(context.llvm_token_type());
    }

    #[test]
    fn test_label_type() {
        let context = Context::new();
        let label_type = context.llvm_label_type();
        assert_eq!(&context, label_type.context());
        assert_eq!(label_type.dialect().namespace().unwrap(), "llvm");
    }

    #[test]
    fn test_label_type_equality() {
        let context = Context::new();
        let label_type_1 = context.llvm_label_type();
        let label_type_2 = context.llvm_label_type();
        assert_eq!(label_type_1, label_type_2);

        let context = Context::new();
        let label_type_2 = context.llvm_label_type();
        assert_ne!(label_type_1, label_type_2);
    }

    #[test]
    fn test_label_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.llvm_label_type(), "!llvm.label");
    }

    #[test]
    fn test_label_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        assert_eq!(context.parse_type("!llvm.label").unwrap(), context.llvm_label_type());
    }

    #[test]
    fn test_label_type_casting() {
        let context = Context::new();
        test_type_casting(context.llvm_label_type());
    }

    #[test]
    fn test_metadata_type() {
        let context = Context::new();
        let metadata_type = context.llvm_metadata_type();
        assert_eq!(&context, metadata_type.context());
        assert_eq!(metadata_type.dialect().namespace().unwrap(), "llvm");
    }

    #[test]
    fn test_metadata_type_equality() {
        let context = Context::new();
        let metadata_type_1 = context.llvm_metadata_type();
        let metadata_type_2 = context.llvm_metadata_type();
        assert_eq!(metadata_type_1, metadata_type_2);

        let context = Context::new();
        let metadata_type_2 = context.llvm_metadata_type();
        assert_ne!(metadata_type_1, metadata_type_2);
    }

    #[test]
    fn test_metadata_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.llvm_metadata_type(), "!llvm.metadata");
    }

    #[test]
    fn test_metadata_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        assert_eq!(context.parse_type("!llvm.metadata").unwrap(), context.llvm_metadata_type());
    }

    #[test]
    fn test_metadata_type_casting() {
        let context = Context::new();
        test_type_casting(context.llvm_metadata_type());
    }

    #[test]
    fn test_array_type() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let array_type = context.llvm_array_type(i32_type, 4);
        assert_eq!(&context, array_type.context());
        assert_eq!(array_type.dialect().namespace().unwrap(), "llvm");
        assert_eq!(array_type.type_id(), ArrayTypeRef::type_id());
        assert_eq!(array_type.element_count(), 4);
        assert_eq!(array_type.element_type(), i32_type);
    }

    #[test]
    fn test_array_type_equality() {
        let context = Context::new();
        let array_type_1 = context.llvm_array_type(context.signless_integer_type(32), 4);
        let array_type_2 = context.llvm_array_type(context.signless_integer_type(32), 4);
        assert_eq!(array_type_1, array_type_2);

        let array_type_2 = context.llvm_array_type(context.signless_integer_type(32), 8);
        assert_ne!(array_type_1, array_type_2);

        let context = Context::new();
        let array_type_2 = context.llvm_array_type(context.signless_integer_type(32), 4);
        assert_ne!(array_type_1, array_type_2);
    }

    #[test]
    fn test_array_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(
            context.llvm_array_type(context.signless_integer_type(32), 4),
            "!llvm.array<4 x i32>",
        );
    }

    #[test]
    fn test_array_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let array_type = context.llvm_array_type(context.signless_integer_type(32), 4);
        assert_eq!(context.parse_type("!llvm.array<4 x i32>").unwrap(), array_type);
    }

    #[test]
    fn test_array_type_casting() {
        let context = Context::new();
        test_type_casting(context.llvm_array_type(context.signless_integer_type(32), 4));
    }

    #[test]
    fn test_function_type() {
        let context = Context::new();
        let pointer_type = context.llvm_pointer_type(0);
        let i32_type = context.signless_integer_type(32);
        let function_type =
            context.llvm_function_type(context.llvm_void_type(), &[pointer_type.as_ref(), i32_type.as_ref()], false);
        assert_eq!(&context, function_type.context());
        assert_eq!(function_type.dialect().namespace().unwrap(), "llvm");
        assert_eq!(function_type.type_id(), FunctionTypeRef::type_id());
        assert_eq!(function_type.input_count(), 2);
        assert!(!function_type.is_variadic());
        assert_eq!(function_type.inputs().collect::<Vec<_>>(), vec![pointer_type.as_ref(), i32_type.as_ref()]);
        assert_eq!(function_type.input(0), pointer_type);
        assert_eq!(function_type.input(1), i32_type);
        assert_eq!(function_type.return_type(), context.llvm_void_type());
    }

    #[test]
    fn test_function_type_equality() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let function_type_1 = context.llvm_function_type(context.llvm_void_type(), &[i32_type], false);
        let function_type_2 = context.llvm_function_type(context.llvm_void_type(), &[i32_type], false);
        assert_eq!(function_type_1, function_type_2);

        let function_type_2 = context.llvm_function_type(i32_type, &[i32_type], false);
        assert_ne!(function_type_1, function_type_2);

        let context = Context::new();
        let function_type_2 =
            context.llvm_function_type(context.llvm_void_type(), &[context.signless_integer_type(32)], false);
        assert_ne!(function_type_1, function_type_2);
    }

    #[test]
    fn test_function_type_display_and_debug() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        test_type_display_and_debug(
            context.llvm_function_type(context.llvm_void_type(), &[i32_type], true),
            "!llvm.func<void (i32, ...)>",
        );
    }

    #[test]
    fn test_function_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let i32_type = context.signless_integer_type(32);
        assert_eq!(
            context.parse_type("!llvm.func<void (i32, ...)>").unwrap(),
            context.llvm_function_type(context.llvm_void_type(), &[i32_type], true),
        );
    }

    #[test]
    fn test_function_type_casting() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        test_type_casting(context.llvm_function_type(context.llvm_void_type(), &[i32_type], false));
    }

    #[test]
    fn test_struct_type() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        let struct_type = context.llvm_literal_struct_type(&[i32_type.as_ref(), pointer_type.as_ref()], true);
        assert_eq!(&context, struct_type.context());
        assert_eq!(struct_type.dialect().namespace().unwrap(), "llvm");
        assert_eq!(struct_type.type_id(), StructTypeRef::type_id());
        assert!(struct_type.is_literal());
        assert!(!struct_type.is_identified());
        assert!(struct_type.is_packed());
        assert!(!struct_type.is_opaque());
        assert_eq!(struct_type.identifier(), None);
        assert_eq!(struct_type.element_count(), Some(2));
        assert_eq!(
            struct_type.element_types().unwrap().collect::<Vec<_>>(),
            vec![i32_type.as_ref(), pointer_type.as_ref()],
        );
        assert_eq!(struct_type.element_type(0), i32_type);
        assert_eq!(struct_type.element_type(1), pointer_type);
    }

    #[test]
    fn test_identified_struct_type() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let struct_type = context.llvm_identified_struct_type("node");
        assert!(struct_type.is_identified());
        assert!(struct_type.is_opaque());
        assert_eq!(struct_type.identifier().unwrap().as_str().unwrap(), "node");
        assert_eq!(struct_type.element_count(), None);

        assert!(struct_type.set_body(&[i32_type], false).is_success());
        assert!(!struct_type.is_opaque());
        assert_eq!(struct_type.element_count(), Some(1));
        assert_eq!(struct_type.element_type(0), i32_type);
    }

    #[test]
    fn test_new_identified_struct_type() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let struct_type = context.llvm_new_identified_struct_type("new_node", &[i32_type], true);
        assert_eq!(&context, struct_type.context());
        assert_eq!(struct_type.dialect().namespace().unwrap(), "llvm");
        assert!(struct_type.is_identified());
        assert!(!struct_type.is_literal());
        assert!(struct_type.is_packed());
        assert!(!struct_type.is_opaque());
        assert_eq!(struct_type.identifier().unwrap().as_str().unwrap(), "new_node");
        assert_eq!(struct_type.element_count(), Some(1));
        assert_eq!(struct_type.element_type(0), i32_type);
    }

    #[test]
    fn test_opaque_struct_type() {
        let context = Context::new();
        let struct_type = context.llvm_opaque_struct_type("opaque_node");
        assert!(struct_type.is_identified());
        assert!(struct_type.is_opaque());
        assert_eq!(struct_type.identifier().unwrap().as_str().unwrap(), "opaque_node");
        assert_eq!(struct_type.element_count(), None);
        assert!(struct_type.set_body(&[context.signless_integer_type(32)], false).is_failure());
    }

    #[test]
    fn test_struct_type_equality() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let struct_type_1 = context.llvm_literal_struct_type(&[i32_type], false);
        let struct_type_2 = context.llvm_literal_struct_type(&[i32_type], false);
        assert_eq!(struct_type_1, struct_type_2);

        let struct_type_2 = context.llvm_literal_struct_type(&[i32_type], true);
        assert_ne!(struct_type_1, struct_type_2);

        let context = Context::new();
        let struct_type_2 = context.llvm_literal_struct_type(&[context.signless_integer_type(32)], false);
        assert_ne!(struct_type_1, struct_type_2);
    }

    #[test]
    fn test_struct_type_display_and_debug() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        test_type_display_and_debug(context.llvm_literal_struct_type(&[i32_type], false), "!llvm.struct<(i32)>");
        test_type_display_and_debug(context.llvm_literal_struct_type(&[i32_type], true), "!llvm.struct<packed (i32)>");
        test_type_display_and_debug(
            context.llvm_opaque_struct_type("opaque_node"),
            "!llvm.struct<\"opaque_node\", opaque>",
        );
    }

    #[test]
    fn test_struct_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let i32_type = context.signless_integer_type(32);
        assert_eq!(
            context.parse_type("!llvm.struct<(i32)>").unwrap(),
            context.llvm_literal_struct_type(&[i32_type], false),
        );
        assert_eq!(
            context.parse_type("!llvm.struct<packed (i32)>").unwrap(),
            context.llvm_literal_struct_type(&[i32_type], true),
        );
    }

    #[test]
    fn test_struct_type_casting() {
        let context = Context::new();
        test_type_casting(context.llvm_literal_struct_type(&[context.signless_integer_type(32)], false));
    }

    #[test]
    fn test_target_ext_type() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let target_ext_type = context.llvm_target_ext_type("target1", &[i32_type.as_ref()], &[1]);
        assert_eq!(&context, target_ext_type.context());
        assert_eq!(target_ext_type.dialect().namespace().unwrap(), "llvm");
        assert_eq!(target_ext_type.name().as_str().unwrap(), "target1");
        assert_eq!(target_ext_type.type_parameter_count(), 1);
        assert_eq!(target_ext_type.type_parameters().collect::<Vec<_>>(), vec![i32_type.as_ref()]);
        assert_eq!(target_ext_type.type_parameter(0), i32_type);
        assert_eq!(target_ext_type.integer_parameter_count(), 1);
        assert_eq!(target_ext_type.integer_parameters().collect::<Vec<_>>(), vec![1]);
        assert_eq!(target_ext_type.integer_parameter(0), 1);
    }

    #[test]
    fn test_target_ext_type_equality() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let target_ext_type_1 = context.llvm_target_ext_type("target1", &[i32_type.as_ref()], &[1]);
        let target_ext_type_2 = context.llvm_target_ext_type("target1", &[i32_type.as_ref()], &[1]);
        assert_eq!(target_ext_type_1, target_ext_type_2);

        let target_ext_type_2 = context.llvm_target_ext_type("target2", &[], &[]);
        assert_ne!(target_ext_type_1, target_ext_type_2);

        let context = Context::new();
        let target_ext_type_2 =
            context.llvm_target_ext_type("target1", &[context.signless_integer_type(32).as_ref()], &[1]);
        assert_ne!(target_ext_type_1, target_ext_type_2);
    }

    #[test]
    fn test_target_ext_type_display_and_debug() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        test_type_display_and_debug(
            context.llvm_target_ext_type("target1", &[i32_type.as_ref()], &[1]),
            "!llvm.target<\"target1\", i32, 1>",
        );
    }

    #[test]
    fn test_target_ext_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let i32_type = context.signless_integer_type(32);
        assert_eq!(
            context.parse_type("!llvm.target<\"target1\", i32, 1>").unwrap(),
            context.llvm_target_ext_type("target1", &[i32_type.as_ref()], &[1]),
        );
    }

    #[test]
    fn test_target_ext_type_casting() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        test_type_casting(context.llvm_target_ext_type("target1", &[i32_type.as_ref()], &[1]));
    }

    #[test]
    fn test_x86_amx_type() {
        let context = Context::new();
        let x86_amx_type = context.llvm_x86_amx_type();
        assert_eq!(&context, x86_amx_type.context());
        assert_eq!(x86_amx_type.dialect().namespace().unwrap(), "llvm");
    }

    #[test]
    fn test_x86_amx_type_equality() {
        let context = Context::new();
        let x86_amx_type_1 = context.llvm_x86_amx_type();
        let x86_amx_type_2 = context.llvm_x86_amx_type();
        assert_eq!(x86_amx_type_1, x86_amx_type_2);

        let context = Context::new();
        let x86_amx_type_2 = context.llvm_x86_amx_type();
        assert_ne!(x86_amx_type_1, x86_amx_type_2);
    }

    #[test]
    fn test_x86_amx_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.llvm_x86_amx_type(), "!llvm.x86_amx");
    }

    #[test]
    fn test_x86_amx_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        assert_eq!(context.parse_type("!llvm.x86_amx").unwrap(), context.llvm_x86_amx_type());
    }

    #[test]
    fn test_x86_amx_type_casting() {
        let context = Context::new();
        test_type_casting(context.llvm_x86_amx_type());
    }

    #[test]
    fn test_ppc_fp128_type() {
        let context = Context::new();
        let ppc_fp128_type = context.llvm_ppc_fp128_type();
        assert_eq!(&context, ppc_fp128_type.context());
        assert_eq!(ppc_fp128_type.dialect().namespace().unwrap(), "llvm");
    }

    #[test]
    fn test_ppc_fp128_type_equality() {
        let context = Context::new();
        let ppc_fp128_type_1 = context.llvm_ppc_fp128_type();
        let ppc_fp128_type_2 = context.llvm_ppc_fp128_type();
        assert_eq!(ppc_fp128_type_1, ppc_fp128_type_2);

        let context = Context::new();
        let ppc_fp128_type_2 = context.llvm_ppc_fp128_type();
        assert_ne!(ppc_fp128_type_1, ppc_fp128_type_2);
    }

    #[test]
    fn test_ppc_fp128_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.llvm_ppc_fp128_type(), "!llvm.ppc_fp128");
    }

    #[test]
    fn test_ppc_fp128_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        assert_eq!(context.parse_type("!llvm.ppc_fp128").unwrap(), context.llvm_ppc_fp128_type());
    }

    #[test]
    fn test_ppc_fp128_type_casting() {
        let context = Context::new();
        test_type_casting(context.llvm_ppc_fp128_type());
    }
}
