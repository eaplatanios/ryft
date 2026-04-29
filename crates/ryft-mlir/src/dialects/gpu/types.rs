use ryft_xla_sys::bindings::{MlirType, mlirGPUAsyncTokenTypeGet};

use crate::{Context, DialectHandle, Type, TypeRef, mlir_subtype_trait_impls};

/// GPU asynchronous token [`Type`]. GPU async tokens order operations that execute asynchronously on the device.
/// Operations implementing the GPU `async` operation interface consume zero or more token dependencies and may produce
/// a new token when launched asynchronously.
///
/// Refer to the [official MLIR GPU dialect documentation](https://mlir.llvm.org/docs/Dialects/GPU/#gpuasynctoken)
/// for more information.
#[derive(Copy, Clone)]
pub struct AsyncTokenTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

mlir_subtype_trait_impls!(
    AsyncTokenTypeRef<'c, 't> as Type,
    mlir_type = Type,
    mlir_subtype = GPUAsyncTokenType,
);

/// Operand role for a GPU MMA matrix fragment.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MmaMatrixOperand {
    /// Left-hand-side multiplicand fragment in `C += A * B`.
    A,

    /// Right-hand-side multiplicand fragment in `C += A * B`.
    B,

    /// Accumulator/result fragment in `C += A * B`.
    C,
}

impl MmaMatrixOperand {
    /// Returns the MLIR spelling used in `!gpu.mma_matrix` types.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::A => "AOp",
            Self::B => "BOp",
            Self::C => "COp",
        }
    }
}

/// GPU MMA matrix [`Type`]. This type represents a matrix fragment held collectively by a subgroup for matrix
/// multiply-accumulate (MMA) operations. The current MLIR C API does not expose a direct constructor/checker, so this
/// wrapper is built through the dialect parser and specialized by checking the printed dialect type spelling.
///
/// Refer to the [official MLIR GPU dialect documentation](https://mlir.llvm.org/docs/Dialects/GPU/#gpummamatrix)
/// for more information.
#[derive(Copy, Clone)]
pub struct MmaMatrixTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> MmaMatrixTypeRef<'c, 't> {
    /// Returns the two-dimensional matrix shape.
    pub fn shape(&self) -> [usize; 2] {
        // TODO(eaplatanios): Replace this printed-form parsing with a proper `ryft-xla-sys` C++ binding for
        //  `gpu::MMAMatrixType` shape accessors.
        let source = self.to_string();
        let payload = source
            .strip_prefix("!gpu.mma_matrix<")
            .and_then(|source| source.strip_suffix(">"))
            .expect("invalid `!gpu.mma_matrix` type");
        let (typed_shape, _) = payload.split_once(", ").expect("invalid `!gpu.mma_matrix` type");
        let (first_dimension, remaining) = typed_shape.split_once('x').expect("invalid `!gpu.mma_matrix` shape");
        let (second_dimension, _) = remaining.rsplit_once('x').expect("invalid `!gpu.mma_matrix` element type");
        [
            first_dimension.parse().expect("invalid `!gpu.mma_matrix` first dimension"),
            second_dimension.parse().expect("invalid `!gpu.mma_matrix` second dimension"),
        ]
    }

    /// Returns the element [`Type`] of this MMA matrix.
    pub fn element_type(&self) -> TypeRef<'c, 't> {
        // TODO(eaplatanios): Replace this printed-form parsing with a proper `ryft-xla-sys` C++ binding for
        //  `gpu::MMAMatrixType` element-type accessors.
        let source = self.to_string();
        let payload = source
            .strip_prefix("!gpu.mma_matrix<")
            .and_then(|source| source.strip_suffix(">"))
            .expect("invalid `!gpu.mma_matrix` type");
        let (typed_shape, _) = payload.split_once(", ").expect("invalid `!gpu.mma_matrix` type");
        let (_, element_type) = typed_shape.rsplit_once('x').expect("invalid `!gpu.mma_matrix` element type");
        self.context().parse_type(element_type).expect("invalid `!gpu.mma_matrix` element type")
    }

    /// Returns the MMA operand role of this matrix.
    pub fn operand(&self) -> MmaMatrixOperand {
        // TODO(eaplatanios): Replace this printed-form parsing with a proper `ryft-xla-sys` C++ binding for
        //  `gpu::MMAMatrixType` operand accessors.
        let source = self.to_string();
        let payload = source
            .strip_prefix("!gpu.mma_matrix<")
            .and_then(|source| source.strip_suffix(">"))
            .expect("invalid `!gpu.mma_matrix` type");
        let (_, operand) = payload.split_once(", ").expect("invalid `!gpu.mma_matrix` operand");
        match operand.trim_matches('"') {
            "AOp" => MmaMatrixOperand::A,
            "BOp" => MmaMatrixOperand::B,
            "COp" => MmaMatrixOperand::C,
            _ => panic!("invalid `!gpu.mma_matrix` operand"),
        }
    }
}

impl<'c, 't> Type<'c, 't> for MmaMatrixTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        let r#type = unsafe { TypeRef::from_c_api(handle, context) }?;
        // TODO(eaplatanios): Replace this printed-form subtype check with a proper `ryft-xla-sys` C++ binding for
        //  checking `gpu::MMAMatrixType`.
        if r#type.to_string().starts_with("!gpu.mma_matrix<") { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(MmaMatrixTypeRef<'c, 't> as Type, mlir_type = Type);

macro_rules! gpu_sparse_handle_type {
    ($name:ident, $context_method:ident, $spelling:literal, $description:literal $(,)*) => {
        #[doc = "GPU sparse "]
        #[doc = $description]
        #[doc = " handle [`Type`]."]
        ///
        /// The current MLIR C API does not expose a direct constructor/checker for this type, so this wrapper is built
        /// through the dialect parser and specialized by checking the printed dialect type spelling.
        #[derive(Copy, Clone)]
        pub struct $name<'c, 't> {
            /// Handle that represents this [`Type`] in the MLIR C API.
            handle: MlirType,

            /// [`Context`] that owns this [`Type`].
            context: &'c Context<'t>,
        }

        impl<'c, 't> Type<'c, 't> for $name<'c, 't> {
            unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
                if handle.ptr.is_null() {
                    return None;
                }
                let r#type = unsafe { TypeRef::from_c_api(handle, context) }?;
                // TODO(eaplatanios): Replace this printed-form subtype check with a proper `ryft-xla-sys` C++
                //  binding for GPU sparse handle types.
                if r#type.to_string() == concat!("!gpu.", $spelling) {
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

        mlir_subtype_trait_impls!($name<'c, 't> as Type, mlir_type = Type);
    };
}

gpu_sparse_handle_type!(
    SparseDnTensorHandleTypeRef,
    gpu_sparse_dn_tensor_handle_type,
    "sparse.dntensor_handle",
    "dense tensor",
);

gpu_sparse_handle_type!(
    SparseSpMatHandleTypeRef,
    gpu_sparse_sp_mat_handle_type,
    "sparse.spmat_handle",
    "sparse matrix",
);

gpu_sparse_handle_type!(
    SparseSpGemmOperationHandleTypeRef,
    gpu_sparse_sp_gemm_operation_handle_type,
    "sparse.spgemmop_handle",
    "SpGEMM operation",
);

impl<'t> Context<'t> {
    /// Creates a new GPU [`AsyncTokenTypeRef`] owned by this [`Context`].
    pub fn gpu_async_token_type<'c>(&'c self) -> AsyncTokenTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        unsafe { AsyncTokenTypeRef::from_c_api(mlirGPUAsyncTokenTypeGet(*self.handle.borrow()), self).unwrap() }
    }

    /// Creates a new GPU [`MmaMatrixTypeRef`] owned by this [`Context`].
    pub fn gpu_mma_matrix_type<'c, T: Type<'c, 't>>(
        &'c self,
        shape: [usize; 2],
        element_type: T,
        operand: MmaMatrixOperand,
    ) -> MmaMatrixTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        // TODO(eaplatanios): Replace this textual construction path with a proper `ryft-xla-sys` C++ binding for
        //  constructing `gpu::MMAMatrixType`.
        self.parse_type(format!(
            "!gpu.mma_matrix<{}x{}x{}, \"{}\">",
            shape[0],
            shape[1],
            element_type,
            operand.as_str(),
        ))
        .and_then(|r#type| r#type.cast())
        .expect("invalid arguments to `Context::gpu_mma_matrix_type`")
    }

    /// Creates a new GPU dense tensor sparse handle type owned by this [`Context`].
    pub fn gpu_sparse_dn_tensor_handle_type<'c>(&'c self) -> SparseDnTensorHandleTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        // TODO(eaplatanios): Replace this textual construction path with a proper `ryft-xla-sys` C++ binding for
        //  constructing `gpu::SparseDnTensorHandleType`.
        self.parse_type("!gpu.sparse.dntensor_handle")
            .and_then(|r#type| r#type.cast())
            .expect("invalid GPU dense tensor sparse handle type")
    }

    /// Creates a new GPU sparse matrix handle type owned by this [`Context`].
    pub fn gpu_sparse_sp_mat_handle_type<'c>(&'c self) -> SparseSpMatHandleTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        // TODO(eaplatanios): Replace this textual construction path with a proper `ryft-xla-sys` C++ binding for
        //  constructing `gpu::SparseSpMatHandleType`.
        self.parse_type("!gpu.sparse.spmat_handle")
            .and_then(|r#type| r#type.cast())
            .expect("invalid GPU sparse matrix handle type")
    }

    /// Creates a new GPU SpGEMM operation handle type owned by this [`Context`].
    pub fn gpu_sparse_sp_gemm_operation_handle_type<'c>(&'c self) -> SparseSpGemmOperationHandleTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        // TODO(eaplatanios): Replace this textual construction path with a proper `ryft-xla-sys` C++ binding for
        //  constructing `gpu::SparseSpGEMMOpHandleType`.
        self.parse_type("!gpu.sparse.spgemmop_handle")
            .and_then(|r#type| r#type.cast())
            .expect("invalid GPU SpGEMM operation handle type")
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Type;
    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_async_token_type() {
        let context = Context::new();
        let token_type = context.gpu_async_token_type();
        assert_eq!(&context, token_type.context());
        assert_eq!(token_type.dialect().namespace().unwrap(), "gpu");
    }

    #[test]
    fn test_async_token_type_equality() {
        let context = Context::new();

        // Token types from the same context must be equal because they are "uniqued".
        let token_type_1 = context.gpu_async_token_type();
        let token_type_2 = context.gpu_async_token_type();
        assert_eq!(token_type_1, token_type_2);

        // Token types from different contexts must not be equal.
        let context = Context::new();
        let token_type_2 = context.gpu_async_token_type();
        assert_ne!(token_type_1, token_type_2);
    }

    #[test]
    fn test_async_token_type_display_and_debug() {
        let context = Context::new();
        let token_type = context.gpu_async_token_type();
        test_type_display_and_debug(token_type, "!gpu.async.token");
    }

    #[test]
    fn test_async_token_type_parsing() {
        let context = Context::new();
        let token_type = context.gpu_async_token_type();
        assert_eq!(context.parse_type("!gpu.async.token").unwrap(), token_type);
    }

    #[test]
    fn test_async_token_type_casting() {
        let context = Context::new();
        let token_type = context.gpu_async_token_type();
        test_type_casting(token_type);
    }

    #[test]
    fn test_mma_matrix_operand() {
        assert_eq!(MmaMatrixOperand::A.as_str(), "AOp");
        assert_eq!(MmaMatrixOperand::B.as_str(), "BOp");
        assert_eq!(MmaMatrixOperand::C.as_str(), "COp");
    }

    #[test]
    fn test_mma_matrix_type() {
        let context = Context::new();
        let mma_type = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::A);
        assert_eq!(&context, mma_type.context());
        assert_eq!(mma_type.dialect().namespace().unwrap(), "gpu");
        assert_eq!(mma_type.shape(), [16, 8]);
        assert_eq!(mma_type.element_type(), context.float32_type());
        assert_eq!(mma_type.operand(), MmaMatrixOperand::A);
    }

    #[test]
    fn test_mma_matrix_type_equality() {
        let context = Context::new();

        // Same types from the same context must be equal because they are "uniqued".
        let mma_type_1 = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::A);
        let mma_type_2 = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::A);
        assert_eq!(mma_type_1, mma_type_2);

        // Different types from the same context must not be equal.
        let mma_type_2 = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::B);
        assert_ne!(mma_type_1, mma_type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let mma_type_2 = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::A);
        assert_ne!(mma_type_1, mma_type_2);
    }

    #[test]
    fn test_mma_matrix_type_display_and_debug() {
        let context = Context::new();
        let mma_type = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::A);
        test_type_display_and_debug(mma_type, "!gpu.mma_matrix<16x8xf32, \"AOp\">");
    }

    #[test]
    fn test_mma_matrix_type_parsing() {
        let context = Context::new();
        let mma_type = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::A);
        assert_eq!(context.parse_type("!gpu.mma_matrix<16x8xf32, \"AOp\">").unwrap(), mma_type);
    }

    #[test]
    fn test_mma_matrix_type_casting() {
        let context = Context::new();
        let mma_type = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::A);
        test_type_casting(mma_type);
    }

    #[test]
    fn test_sparse_dn_tensor_handle_type() {
        let context = Context::new();
        let handle_type = context.gpu_sparse_dn_tensor_handle_type();
        assert_eq!(&context, handle_type.context());
        assert_eq!(handle_type.dialect().namespace().unwrap(), "gpu");
    }

    #[test]
    fn test_sparse_dn_tensor_handle_type_equality() {
        let context = Context::new();

        // Sparse handle types from the same context must be equal because they are "uniqued".
        let handle_type_1 = context.gpu_sparse_dn_tensor_handle_type();
        let handle_type_2 = context.gpu_sparse_dn_tensor_handle_type();
        assert_eq!(handle_type_1, handle_type_2);

        // Sparse handle types from different contexts must not be equal.
        let context = Context::new();
        let handle_type_2 = context.gpu_sparse_dn_tensor_handle_type();
        assert_ne!(handle_type_1, handle_type_2);
    }

    #[test]
    fn test_sparse_dn_tensor_handle_type_display_and_debug() {
        let context = Context::new();
        let handle_type = context.gpu_sparse_dn_tensor_handle_type();
        test_type_display_and_debug(handle_type, "!gpu.sparse.dntensor_handle");
    }

    #[test]
    fn test_sparse_dn_tensor_handle_type_parsing() {
        let context = Context::new();
        let handle_type = context.gpu_sparse_dn_tensor_handle_type();
        assert_eq!(context.parse_type("!gpu.sparse.dntensor_handle").unwrap(), handle_type);
    }

    #[test]
    fn test_sparse_dn_tensor_handle_type_casting() {
        let context = Context::new();
        let handle_type = context.gpu_sparse_dn_tensor_handle_type();
        test_type_casting(handle_type);
    }

    #[test]
    fn test_sparse_sp_mat_handle_type() {
        let context = Context::new();
        let handle_type = context.gpu_sparse_sp_mat_handle_type();
        assert_eq!(&context, handle_type.context());
        assert_eq!(handle_type.dialect().namespace().unwrap(), "gpu");
    }

    #[test]
    fn test_sparse_sp_mat_handle_type_equality() {
        let context = Context::new();

        // Sparse handle types from the same context must be equal because they are "uniqued".
        let handle_type_1 = context.gpu_sparse_sp_mat_handle_type();
        let handle_type_2 = context.gpu_sparse_sp_mat_handle_type();
        assert_eq!(handle_type_1, handle_type_2);

        // Sparse handle types from different contexts must not be equal.
        let context = Context::new();
        let handle_type_2 = context.gpu_sparse_sp_mat_handle_type();
        assert_ne!(handle_type_1, handle_type_2);
    }

    #[test]
    fn test_sparse_sp_mat_handle_type_display_and_debug() {
        let context = Context::new();
        let handle_type = context.gpu_sparse_sp_mat_handle_type();
        test_type_display_and_debug(handle_type, "!gpu.sparse.spmat_handle");
    }

    #[test]
    fn test_sparse_sp_mat_handle_type_parsing() {
        let context = Context::new();
        let handle_type = context.gpu_sparse_sp_mat_handle_type();
        assert_eq!(context.parse_type("!gpu.sparse.spmat_handle").unwrap(), handle_type);
    }

    #[test]
    fn test_sparse_sp_mat_handle_type_casting() {
        let context = Context::new();
        let handle_type = context.gpu_sparse_sp_mat_handle_type();
        test_type_casting(handle_type);
    }

    #[test]
    fn test_sparse_sp_gemm_operation_handle_type() {
        let context = Context::new();
        let handle_type = context.gpu_sparse_sp_gemm_operation_handle_type();
        assert_eq!(&context, handle_type.context());
        assert_eq!(handle_type.dialect().namespace().unwrap(), "gpu");
    }

    #[test]
    fn test_sparse_sp_gemm_operation_handle_type_equality() {
        let context = Context::new();

        // Sparse handle types from the same context must be equal because they are "uniqued".
        let handle_type_1 = context.gpu_sparse_sp_gemm_operation_handle_type();
        let handle_type_2 = context.gpu_sparse_sp_gemm_operation_handle_type();
        assert_eq!(handle_type_1, handle_type_2);

        // Sparse handle types from different contexts must not be equal.
        let context = Context::new();
        let handle_type_2 = context.gpu_sparse_sp_gemm_operation_handle_type();
        assert_ne!(handle_type_1, handle_type_2);
    }

    #[test]
    fn test_sparse_sp_gemm_operation_handle_type_display_and_debug() {
        let context = Context::new();
        let handle_type = context.gpu_sparse_sp_gemm_operation_handle_type();
        test_type_display_and_debug(handle_type, "!gpu.sparse.spgemmop_handle");
    }

    #[test]
    fn test_sparse_sp_gemm_operation_handle_type_parsing() {
        let context = Context::new();
        let handle_type = context.gpu_sparse_sp_gemm_operation_handle_type();
        assert_eq!(context.parse_type("!gpu.sparse.spgemmop_handle").unwrap(), handle_type);
    }

    #[test]
    fn test_sparse_sp_gemm_operation_handle_type_casting() {
        let context = Context::new();
        let handle_type = context.gpu_sparse_sp_gemm_operation_handle_type();
        test_type_casting(handle_type);
    }
}
