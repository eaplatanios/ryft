//! The TritonGPU `ttg` dialect describes workgroup-level execution and memory operations.

use ryft_xla_sys::mlir::dialects::triton::ttg::mlirGetDialectHandle__ttg__;

use crate::{DialectHandle, Error};

pub mod attributes;
pub mod operations;

pub use attributes::*;
pub use operations::*;

impl DialectHandle<'_, '_> {
    /// Returns the TritonGPU dialect handle when the linked native archive includes TritonGPU.
    pub fn triton_ttg() -> Result<Self, Error> {
        let handle = unsafe { mlirGetDialectHandle__ttg__() };
        if handle.ptr.is_null() {
            return Err(Error::internal("the linked native archive does not include the TritonGPU dialect"));
        }
        unsafe { Self::from_c_api(handle) }
    }
}
