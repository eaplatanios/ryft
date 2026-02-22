// TODO(eaplatanios): Clean this up and make sure it is correct.

use ryft_xla_sys::bindings::{MlirType, mlirGPUAsyncTokenTypeGet, mlirTypeIsAGPUAsyncTokenType};

use crate::{Context, DialectHandle, OpaqueTypeRef, Type, TypeRef, mlir_subtype_trait_impls};

/// MLIR [`Type`] wrapper for types that belong to the `gpu` dialect namespace.
#[derive(Copy, Clone)]
pub struct GpuTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> GpuTypeRef<'c, 't> {
    /// Returns the type mnemonic if it can be derived from its textual form.
    pub fn mnemonic(&self) -> Option<String> {
        type_mnemonic(*self)
    }
}

impl<'c, 't> Type<'c, 't> for GpuTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        let type_ref = unsafe { TypeRef::from_c_api(handle, context) }?;
        if type_is_gpu(type_ref) { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(GpuTypeRef<'c, 't> as Type, mlir_type = Type);

/// `gpu.async.token` [`Type`].
#[derive(Copy, Clone)]
pub struct AsyncTokenTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for AsyncTokenTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAGPUAsyncTokenType(handle) } {
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

mlir_subtype_trait_impls!(AsyncTokenTypeRef<'c, 't> as Type, mlir_type = Type);

/// `gpu.mma_matrix` [`Type`].
#[derive(Copy, Clone)]
pub struct MmaMatrixTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> MmaMatrixTypeRef<'c, 't> {
    /// Returns the matrix shape if it can be parsed from the textual form.
    pub fn shape(&self) -> Option<Vec<i64>> {
        parse_mma_matrix_type(self.to_string()).map(|(shape, _, _)| shape)
    }

    /// Returns the element [`Type`] if it can be parsed from the textual form.
    pub fn element_type(&self) -> Option<TypeRef<'c, 't>> {
        parse_mma_matrix_type(self.to_string())
            .and_then(|(_, element_type, _)| self.context.parse_type(element_type).map(|parsed| parsed))
    }

    /// Returns the operand designator (e.g., `AOp`, `BOp`, or `COp`) if present.
    pub fn operand(&self) -> Option<String> {
        parse_mma_matrix_type(self.to_string()).map(|(_, _, operand)| operand)
    }
}

impl<'c, 't> Type<'c, 't> for MmaMatrixTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        let type_ref = unsafe { TypeRef::from_c_api(handle, context) }?;
        if type_mnemonic(type_ref).as_deref() == Some("mma_matrix") { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(MmaMatrixTypeRef<'c, 't> as Type, mlir_type = Type);

/// `gpu.sparse.dntensor_handle` [`Type`].
#[derive(Copy, Clone)]
pub struct SparseDnTensorHandleTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for SparseDnTensorHandleTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        let type_ref = unsafe { TypeRef::from_c_api(handle, context) }?;
        if type_mnemonic(type_ref).as_deref() == Some("sparse.dntensor_handle") {
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

mlir_subtype_trait_impls!(SparseDnTensorHandleTypeRef<'c, 't> as Type, mlir_type = Type);

/// `gpu.sparse.spmat_handle` [`Type`].
#[derive(Copy, Clone)]
pub struct SparseSpMatHandleTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for SparseSpMatHandleTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        let type_ref = unsafe { TypeRef::from_c_api(handle, context) }?;
        if type_mnemonic(type_ref).as_deref() == Some("sparse.spmat_handle") {
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

mlir_subtype_trait_impls!(SparseSpMatHandleTypeRef<'c, 't> as Type, mlir_type = Type);

/// `gpu.sparse.spgemmop_handle` [`Type`].
#[derive(Copy, Clone)]
pub struct SparseSpGemmOpHandleTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for SparseSpGemmOpHandleTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        let type_ref = unsafe { TypeRef::from_c_api(handle, context) }?;
        if type_mnemonic(type_ref).as_deref() == Some("sparse.spgemmop_handle") {
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

mlir_subtype_trait_impls!(SparseSpGemmOpHandleTypeRef<'c, 't> as Type, mlir_type = Type);

impl<'t> Context<'t> {
    /// Parses a GPU dialect type from the provided source string.
    pub fn parse_gpu_type<'c, S: AsRef<str>>(&'c self, source: S) -> Option<GpuTypeRef<'c, 't>> {
        self.load_dialect(DialectHandle::gpu());
        self.parse_type(source.as_ref()).and_then(|r#type| r#type.cast::<GpuTypeRef>())
    }

    /// Creates a new `gpu.async.token` [`Type`] owned by this [`Context`].
    pub fn gpu_async_token_type<'c>(&'c self) -> AsyncTokenTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        unsafe { AsyncTokenTypeRef::from_c_api(mlirGPUAsyncTokenTypeGet(*self.handle.borrow()), &self).unwrap() }
    }

    /// Creates a new `gpu.mma_matrix` [`Type`] owned by this [`Context`].
    pub fn gpu_mma_matrix_type<'c, T: Type<'c, 't>>(
        &'c self,
        shape: &[i64],
        element_type: T,
        operand: &str,
    ) -> MmaMatrixTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        let mut shape_string = shape.iter().map(|dim| dim.to_string()).collect::<Vec<_>>().join("x");
        shape_string.push('x');
        shape_string.push_str(&element_type.to_string());
        let rendered = format!("!gpu.mma_matrix<{shape_string}, \"{operand}\">");
        self.parse_type(rendered)
            .and_then(|r#type| r#type.cast::<MmaMatrixTypeRef>())
            .expect("invalid arguments to `Context::gpu_mma_matrix_type`")
    }

    /// Creates a new `gpu.sparse.dntensor_handle` [`Type`] owned by this [`Context`].
    pub fn gpu_sparse_dn_tensor_handle_type<'c>(&'c self) -> SparseDnTensorHandleTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        self.parse_type("!gpu.sparse.dntensor_handle")
            .and_then(|r#type| r#type.cast::<SparseDnTensorHandleTypeRef>())
            .expect("invalid gpu sparse dn tensor handle type")
    }

    /// Creates a new `gpu.sparse.spmat_handle` [`Type`] owned by this [`Context`].
    pub fn gpu_sparse_spmat_handle_type<'c>(&'c self) -> SparseSpMatHandleTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        self.parse_type("!gpu.sparse.spmat_handle")
            .and_then(|r#type| r#type.cast::<SparseSpMatHandleTypeRef>())
            .expect("invalid gpu sparse spmat handle type")
    }

    /// Creates a new `gpu.sparse.spgemmop_handle` [`Type`] owned by this [`Context`].
    pub fn gpu_sparse_spgemm_op_handle_type<'c>(&'c self) -> SparseSpGemmOpHandleTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        self.parse_type("!gpu.sparse.spgemmop_handle")
            .and_then(|r#type| r#type.cast::<SparseSpGemmOpHandleTypeRef>())
            .expect("invalid gpu sparse spgemm op handle type")
    }
}

fn type_is_gpu<'c, 't: 'c, T: Type<'c, 't>>(r#type: T) -> bool {
    r#type.dialect().namespace().ok().map(|namespace| namespace == "gpu").unwrap_or(false)
        || r#type
            .cast::<OpaqueTypeRef>()
            .and_then(|opaque_type| opaque_type.dialect_namespace().ok().map(|namespace| namespace == "gpu"))
            .unwrap_or(false)
}

fn type_mnemonic<'c, 't: 'c, T: Type<'c, 't>>(r#type: T) -> Option<String> {
    let rendered = r#type.to_string();
    let prefix = "!gpu.";
    if !rendered.starts_with(prefix) {
        return None;
    }

    let suffix = &rendered[prefix.len()..];
    let end = suffix
        .char_indices()
        .find_map(|(index, character)| {
            if character == '<' || character == ' ' || character == '\t' || character == '\n' || character == ':' {
                Some(index)
            } else {
                None
            }
        })
        .unwrap_or(suffix.len());

    Some(suffix[..end].to_owned())
}

fn parse_mma_matrix_type(rendered: String) -> Option<(Vec<i64>, String, String)> {
    let rendered = rendered.trim().to_owned();
    let prefix = "!gpu.mma_matrix<";
    let contents = rendered.strip_prefix(prefix)?.strip_suffix('>')?;
    let mut parts = contents.splitn(2, ',');
    let shape_and_element = parts.next()?.trim();
    let operand = parts.next()?.trim().trim_matches('"').to_owned();
    let mut tokens = shape_and_element.split('x').collect::<Vec<_>>();
    if tokens.len() < 2 {
        return None;
    }
    let element_type = tokens.pop()?.to_owned();
    let shape = tokens.iter().map(|token| token.parse::<i64>().ok()).collect::<Option<Vec<_>>>()?;
    Some((shape, element_type, operand))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_gpu_async_token_type() {
        let context = Context::new();
        let token_type = context.gpu_async_token_type();
        assert_eq!(&context, token_type.context());
        assert_eq!(token_type.dialect().namespace().unwrap(), "gpu");
        test_type_display_and_debug(token_type, "!gpu.async.token");
        test_type_casting(token_type);
    }

    #[test]
    fn test_gpu_type_mnemonic() {
        let context = Context::new();
        let token_type = context.gpu_async_token_type();
        let gpu_type = token_type.cast::<GpuTypeRef>().unwrap();
        assert_eq!(gpu_type.mnemonic().as_deref(), Some("async.token"));
    }

    #[test]
    fn test_gpu_mma_matrix_type() {
        let context = Context::new();
        let element_type = context.float16_type();
        let mma_type = context.gpu_mma_matrix_type(&[16, 16], element_type, "AOp");
        assert_eq!(mma_type.element_type().unwrap(), element_type.as_ref());
        assert_eq!(mma_type.shape().unwrap(), vec![16, 16]);
        assert_eq!(mma_type.operand().as_deref(), Some("AOp"));
        assert_eq!(mma_type.dialect().namespace().unwrap(), "gpu");
    }

    #[test]
    fn test_gpu_sparse_handle_types() {
        let context = Context::new();
        let dn_tensor = context.gpu_sparse_dn_tensor_handle_type();
        let spmat = context.gpu_sparse_spmat_handle_type();
        let spgemm = context.gpu_sparse_spgemm_op_handle_type();
        assert_eq!(dn_tensor.dialect().namespace().unwrap(), "gpu");
        assert_eq!(spmat.dialect().namespace().unwrap(), "gpu");
        assert_eq!(spgemm.dialect().namespace().unwrap(), "gpu");
        assert_eq!(dn_tensor.to_string(), "!gpu.sparse.dntensor_handle");
        assert_eq!(spmat.to_string(), "!gpu.sparse.spmat_handle");
        assert_eq!(spgemm.to_string(), "!gpu.sparse.spgemmop_handle");
    }

    #[test]
    fn test_parse_gpu_type() {
        let context = Context::new();
        let parsed = context.parse_gpu_type("!gpu.async.token");
        assert!(parsed.is_some());
    }
}
