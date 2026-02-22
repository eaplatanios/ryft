#![allow(clippy::missing_safety_doc)]

use std::sync::{Mutex, OnceLock};

use ryft_xla_sys::bindings::mlirRegisterAllPasses;

pub mod attributes;
pub mod blocks;
pub mod context;
pub mod diagnostics;
pub mod dialects;
pub mod execution_engine;
pub mod identifier;
pub mod locations;
pub mod modules;
pub mod operations;
pub mod passes;
pub mod regions;
pub mod support;
pub mod types;
pub mod values;

#[macro_use]
pub mod macros;

pub use self::attributes::*;
pub use self::blocks::*;
pub use self::context::*;
pub use self::diagnostics::*;
pub use self::dialects::affine::{
    AddAffineExpressionRef, AffineExpression, AffineExpressionRef, AffineMap, BinaryOperationAffineExpressionRef,
    CeilDivAffineExpressionRef, ConstantAffineExpressionRef, DimensionAffineExpressionRef, FloorDivAffineExpressionRef,
    IntegerSet, IntegerSetConstraint, ModAffineExpressionRef, MulAffineExpressionRef, SymbolAffineExpressionRef,
};
pub use self::dialects::builtin::{
    AffineMapAttributeRef, ArrayAttributeRef, BFloat16TypeRef, BooleanAttributeRef, CallSiteLocationRef,
    ComplexTypeRef, DenseArrayAttribute, DenseArrayAttributeRef, DenseBooleanArrayAttributeRef,
    DenseElementsAttributeRef, DenseFloat32ArrayAttributeRef, DenseFloat64ArrayAttributeRef,
    DenseFloatElementsAttributeRef, DenseInteger8ArrayAttributeRef, DenseInteger16ArrayAttributeRef,
    DenseInteger32ArrayAttributeRef, DenseInteger64ArrayAttributeRef, DenseIntegerElementsAttributeRef,
    DenseResourceElementsAttributeRef, DetachedModuleOperation, DetachedUnrealizedConversionCastOperation,
    DictionaryAttributeRef, DistinctAttributeRef, ElementsAttribute, ElementsAttributeRef, FileLocationRef,
    FlatSymbolRefAttributeRef, Float4E2M1FNTypeRef, Float6E2M3FNTypeRef, Float6E3M2FNTypeRef, Float8E3M4TypeRef,
    Float8E4M3B11FNUZTypeRef, Float8E4M3FNTypeRef, Float8E4M3FNUZTypeRef, Float8E4M3TypeRef, Float8E5M2FNUZTypeRef,
    Float8E5M2TypeRef, Float8E8M0FNUTypeRef, Float8Type, Float8TypeRef, Float16TypeRef, Float32TypeRef, Float64TypeRef,
    FloatAttributeRef, FloatTF32TypeRef, FloatType, FloatTypeRef, FunctionTypeRef, FusedLocationRef, IndexTypeRef,
    IntegerAttributeRef, IntegerSetAttributeRef, IntegerTypeRef, LocationAttributeRef, MemRefTypeRef, ModuleOperation,
    ModuleOperationRef, NamedLocationRef, NoneTypeRef, OpaqueAttributeRef, OpaqueTypeRef, ShapedType, ShapedTypeRef,
    Size, SparseElementsAttributeRef, StridedLayoutAttributeRef, StringAttributeRef, SymbolRefAttributeRef,
    SymbolVisibilityAttributeRef, TensorTypeRef, TupleTypeRef, TypeAttributeRef, UnitAttributeRef, UnknownLocationRef,
    UnrankedMemRefTypeRef, UnrankedTensorTypeRef, UnrealizedConversionCastOperation,
    UnrealizedConversionCastOperationRef, VectorTypeDimension, VectorTypeRef,
};
pub use self::dialects::{Dialect, DialectHandle, DialectRegistry};
pub use self::execution_engine::*;
pub use self::identifier::*;
pub use self::locations::*;
pub use self::modules::*;
pub use self::operations::*;
pub use self::passes::{ClosurePass, ExternalPass, OperationPassManager, Pass, PassIrPrintingOptions, PassManager};
pub use self::regions::*;
pub use self::support::*;
pub use self::types::*;
pub use self::values::*;

/// Static [`Mutex`] that is used for guarding MLIR operations that are global and not specific to a [`Context`]
/// (e.g., registering dialects or passes which are not thread safe).
pub static GLOBAL_REGISTRATION_MUTEX: Mutex<()> = Mutex::new(());

/// Registers all compiler passes of MLIR (including all [StableHLO](https://openxla.org/stablehlo) and
/// [Shardy](https://openxla.org/shardy) compiler passes) with the global registry. If you are building a compiler,
/// you likely do not need to call this function; in that case, you would build a pipeline programmatically without
/// the need to register with the global registry, since it would already be calling the creation routine of the
/// individual passes. The global registry is interesting when interacting with the MLIR command-line tools.
pub fn register_all_passes() {
    // Use [`OnceLock`] to ensure that [`register_all_passes`] is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterAllPasses()
    });
    dialects::mhlo::register_mhlo_passes();
    dialects::stable_hlo::register_stable_hlo_passes();
    dialects::shardy::register_shardy_passes_and_pipelines();
    dialects::shardy::register_xla_shardy_passes_and_pipelines();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_all_dialects() {
        // We intentionally try to register multiple times to ensure that the operation is idempotent.
        let registry = DialectRegistry::new();
        for _ in 0..1000 {
            registry.insert_all_built_in_dialects();
        }
    }

    #[test]
    fn test_register_all_passes() {
        // We intentionally try to register multiple times in parallel to ensure that the operation is idempotent.
        let threads = (0..100).map(|_| std::thread::spawn(|| register_all_passes())).collect::<Vec<_>>();
        threads.into_iter().for_each(|thread| drop(thread.join()));
    }
}
