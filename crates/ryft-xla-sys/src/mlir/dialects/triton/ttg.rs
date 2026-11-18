use crate::bindings::{MlirAttribute, MlirContext, MlirDialectHandle, MlirType};

unsafe extern "C" {
    /// Returns the TritonGPU dialect handle, or a null handle for an archive without the dialect.
    pub fn mlirGetDialectHandle__ttg__() -> MlirDialectHandle;

    /// Constructs the TritonGPU barrier address-space attribute, returning null for an invalid mask.
    pub fn mlirTritonTtgBarrierAddressSpacesAttrGet(context: MlirContext, spaces: u32) -> MlirAttribute;

    /// Constructs the TritonGPU barrier scope attribute, returning null for an invalid scope.
    pub fn mlirTritonTtgBarrierScopeAttrGet(context: MlirContext, scope: u32) -> MlirAttribute;

    /// Constructs the token type used by TritonGPU asynchronous copies and their group operations.
    pub fn mlirTritonTtgAsyncTokenTypeGet(context: MlirContext) -> MlirType;

    /// Constructs a mutable, unswizzled row-major shared-memory descriptor for one CTA.
    /// The shape must contain `rank` positive dimensions. The element type must belong to a context with
    /// TritonGPU loaded.
    pub fn mlirTritonTtgMutableSharedMemDescTypeGet(element_type: MlirType, rank: isize, shape: *const i64)
    -> MlirType;
}
