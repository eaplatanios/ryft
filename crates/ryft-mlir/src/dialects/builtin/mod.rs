//! The `builtin` dialect contains a core set of [`Attribute`](crate::Attribute)s, [`Operation`](crate::Operation)s,
//! [`Location`](crate::Location)s, and [`Type`](crate::Type)s that have wide applicability across a very large number
//! of domains and abstractions. Many of the components of this dialect are also instrumental in the implementation of
//! the core IR. As such, this dialect is implicitly loaded in every [`Context`](crate::Context), and available
//! directly to all users of MLIR.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/) for more information.

pub mod attributes;
pub mod locations;
pub mod operations;
pub mod passes;
pub mod types;

pub use attributes::{
    AffineMapAttributeRef, ArrayAttributeRef, BooleanAttributeRef, DenseArrayAttribute, DenseArrayAttributeRef,
    DenseBooleanArrayAttributeRef, DenseElementsAttributeRef, DenseFloat32ArrayAttributeRef,
    DenseFloat64ArrayAttributeRef, DenseFloatElementsAttributeRef, DenseInteger8ArrayAttributeRef,
    DenseInteger16ArrayAttributeRef, DenseInteger32ArrayAttributeRef, DenseInteger64ArrayAttributeRef,
    DenseIntegerElementsAttributeRef, DenseResourceElementsAttributeRef, DictionaryAttributeRef, DistinctAttributeRef,
    ElementsAttribute, ElementsAttributeRef, FlatSymbolRefAttributeRef, FloatAttributeRef, IntegerAttributeRef,
    IntegerSetAttributeRef, LocationAttributeRef, OpaqueAttributeRef, SparseElementsAttributeRef,
    StridedLayoutAttributeRef, StringAttributeRef, SymbolRefAttributeRef, SymbolVisibilityAttributeRef,
    TypeAttributeRef, UnitAttributeRef,
};
pub use locations::{CallSiteLocationRef, FileLocationRef, FusedLocationRef, NamedLocationRef, UnknownLocationRef};
pub use operations::{
    DetachedModuleOperation, DetachedUnrealizedConversionCastOperation, ModuleOperation, ModuleOperationRef,
    UnrealizedConversionCastOperation, UnrealizedConversionCastOperationRef, module, named_module,
    unrealized_conversion_cast,
};
pub use passes::*;
pub use types::{
    BFloat16TypeRef, ComplexTypeRef, Float4E2M1FNTypeRef, Float6E2M3FNTypeRef, Float6E3M2FNTypeRef, Float8E3M4TypeRef,
    Float8E4M3B11FNUZTypeRef, Float8E4M3FNTypeRef, Float8E4M3FNUZTypeRef, Float8E4M3TypeRef, Float8E5M2FNUZTypeRef,
    Float8E5M2TypeRef, Float8E8M0FNUTypeRef, Float8Type, Float8TypeRef, Float16TypeRef, Float32TypeRef, Float64TypeRef,
    FloatTF32TypeRef, FloatType, FloatTypeRef, FunctionTypeRef, IndexTypeRef, IntegerTypeRef, MemRefTypeRef,
    NoneTypeRef, OpaqueTypeRef, ShapedType, ShapedTypeRef, Size, TensorTypeRef, TupleTypeRef, UnrankedMemRefTypeRef,
    UnrankedTensorTypeRef, VectorTypeDimension, VectorTypeRef,
};
