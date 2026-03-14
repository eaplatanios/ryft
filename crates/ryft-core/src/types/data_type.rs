use std::backtrace::Backtrace;
use std::fmt::Display;

use ryft_macros::Parameter;
use thiserror::Error;

#[cfg(feature = "xla")]
use ryft_pjrt::BufferType;

use crate::parameters::Parameter;
use crate::types::Type;

/// Represents [`DataType`]-related errors.
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum DataTypeError {
    #[error("{message}")]
    InvalidDataType { message: String, backtrace: String },

    #[error("cannot construct a promoted data type from an empty collection of data types")]
    EmptyDataTypePromotionInput { backtrace: String },

    #[error("{message}")]
    InvalidDataTypePromotion { message: String, backtrace: String },
}

impl DataTypeError {
    /// Creates a new [`DataTypeError::InvalidDataType`].
    pub fn invalid_data_type<M: Into<String>>(message: M) -> Self {
        Self::InvalidDataType { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`DataTypeError::EmptyDataTypePromotionInput`].
    pub fn empty_data_type_promotion_input() -> Self {
        Self::EmptyDataTypePromotionInput { backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`DataTypeError::InvalidDataTypePromotion`].
    pub fn invalid_data_type_promotion<M: Into<String>>(message: M) -> Self {
        Self::InvalidDataTypePromotion { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }
}

#[cfg_attr(doc, aquamarine::aquamarine)]
/// Represents a primitive data type that can be stored in arrays, tensors, matrices, vectors, scalars, etc., which
/// range from standard numeric types including booleans, integers, floating-point numbers, and complex numbers of
/// various precisions to advanced data types that mirror [LLVM/MLIR types](https://mlir.llvm.org/docs/Dialects/Builtin)
/// like [8-bit floating-point variants](https://arxiv.org/abs/2209.05433).
///
/// # Type Promotion
///
/// The data types form an automatic promotion lattice that is used whenever values of multiple types participate in the
/// same operation. The diagram below shows the connected core of that lattice. Each node is a concrete [`DataType`]
/// except for the special `f*/c*` node that is described below the diagram. A solid arrow `A -> B` means that `A`
/// can be promoted directly to `B`, and any longer solid path represents transitive widening to a more general type.
/// The semantics of dashed arrows are explained later along with the semantics of the special `f*/c*` node.
///
/// ```mermaid
///   flowchart LR
///     boolean["bool"]
///
///     i1["i1"] --> i2["i2"] --> i4["i4"] --> i8["i8"] --> i16["i16"] --> i32["i32"] --> i64["i64"]
///     u1["u1"] --> u2["u2"] --> u4["u4"] --> u8["u8"] --> u16["u16"] --> u32["u32"] --> u64["u64"]
///     join["f*/c*"]
///     join -.-> bf16["bf16"] --> f32["f32"]
///     join -.-> f16["f16"] --> f32
///     f32 --> f64["f64"] --> c128["c128"]
///     f32 --> c64["c64"] --> c128
///
///     boolean --> u1
///     u1 --> i2
///     u2 --> i4
///     u4 --> i8
///
///     i64 -.-> join
///     u64 -.-> join
/// ```
///
/// The special `f*/c*` node does not correspond to a real [`DataType`] variant. Instead, it represents some standard
/// floating-point or complex type that is already present in the expression involving the type that is being promoted.
/// Dashed edges from `i64` and `u64` into that node indicate that these wide integer types do not move onto the
/// floating-point or complex side of the lattice on their own, but they may do so when they are combined with an
/// operand that is already on that side. Once that happens, the final result is determined by following the solid
/// arrows on that branch starting at the type of that operand. When only `i64` and `u64` are involved, the result
/// defaults to [`DataType::F64`] because no wider integer type exists.
///
/// Note that the diagram intentionally omits [`DataType::Token`] and the 4-bit and 8-bit floating-point types. That is
/// because those types do not support automatic promotion at all and explicit casting should be used instead, when
/// necessary.
///
/// The matrix below shows the result of [`DataType::promoted`] for each pair of [`DataType`]s that support promotion.
/// Note that it also intentionally omits [`DataType::Token`] and the 4-bit and 8-bit floating-point types as only the
/// diagonal entries would be populated for those.
///
/// <style>
/// .tp-matrix-wrap {
///   overflow-x: auto;
///   margin: 0.75rem 0;
/// }
/// .tp-matrix {
///   border-collapse: collapse;
///   font-family: var(--font-family-code);
///   font-size: 0.92em;
///   white-space: nowrap;
/// }
/// .tp-matrix th,
/// .tp-matrix td {
///   padding: 0.35rem 0.5rem;
///   border: 1px solid var(--border-color);
///   text-align: center;
/// }
/// .tp-matrix th {
///   background: var(--table-alt-row-background-color);
///   font-weight: 600;
/// }
/// .tp-matrix td {
///   font-weight: 600;
/// }
/// .tp-bool {
///   color: var(--keyword-link-color);
/// }
/// .tp-int {
///   color: var(--type-link-color);
/// }
/// .tp-float {
///   color: var(--function-link-color);
/// }
/// .tp-complex {
///   color: var(--macro-link-color);
/// }
/// .tp-matrix td.tp-bool {
///   background: color-mix(
///     in srgb,
///     var(--keyword-link-color) 12%,
///     var(--main-background-color)
///   );
/// }
/// .tp-matrix td.tp-int {
///   background: color-mix(
///     in srgb,
///     var(--type-link-color) 12%,
///     var(--main-background-color)
///   );
/// }
/// .tp-matrix td.tp-float {
///   background: color-mix(
///     in srgb,
///     var(--function-link-color) 12%,
///     var(--main-background-color)
///   );
/// }
/// .tp-matrix td.tp-complex {
///   background: color-mix(
///     in srgb,
///     var(--macro-link-color) 12%,
///     var(--main-background-color)
///   );
/// }
/// .tp-matrix td.tp-na {
///   background: var(--table-alt-row-background-color);
///   color: var(--right-side-color);
///   font-weight: 400;
/// }
/// </style>
///
/// <div class="tp-matrix-wrap">
///   <table class="tp-matrix">
///     <thead>
///       <tr>
///         <th>+</th>
///         <th class="tp-bool">bool</th>
///         <th class="tp-int">i1</th>
///         <th class="tp-int">i2</th>
///         <th class="tp-int">i4</th>
///         <th class="tp-int">i8</th>
///         <th class="tp-int">i16</th>
///         <th class="tp-int">i32</th>
///         <th class="tp-int">i64</th>
///         <th class="tp-int">u1</th>
///         <th class="tp-int">u2</th>
///         <th class="tp-int">u4</th>
///         <th class="tp-int">u8</th>
///         <th class="tp-int">u16</th>
///         <th class="tp-int">u32</th>
///         <th class="tp-int">u64</th>
///         <th class="tp-float">bf16</th>
///         <th class="tp-float">f16</th>
///         <th class="tp-float">f32</th>
///         <th class="tp-float">f64</th>
///         <th class="tp-complex">c64</th>
///         <th class="tp-complex">c128</th>
///       </tr>
///     </thead>
///     <tbody>
///       <tr>
///         <th class="tp-bool" scope="row">bool</th>
///         <td class="tp-bool">bool</td>
///         <td class="tp-int">i2</td>
///         <td class="tp-int">i2</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">u1</td>
///         <td class="tp-int">u2</td>
///         <td class="tp-int">u4</td>
///         <td class="tp-int">u8</td>
///         <td class="tp-int">u16</td>
///         <td class="tp-int">u32</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">i1</th>
///         <td class="tp-int">i2</td>
///         <td class="tp-int">i1</td>
///         <td class="tp-int">i2</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i2</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">i2</th>
///         <td class="tp-int">i2</td>
///         <td class="tp-int">i2</td>
///         <td class="tp-int">i2</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i2</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">i4</th>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">i8</th>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">i16</th>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">i32</th>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">i64</th>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">u1</th>
///         <td class="tp-int">u1</td>
///         <td class="tp-int">i2</td>
///         <td class="tp-int">i2</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">u1</td>
///         <td class="tp-int">u2</td>
///         <td class="tp-int">u4</td>
///         <td class="tp-int">u8</td>
///         <td class="tp-int">u16</td>
///         <td class="tp-int">u32</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">u2</th>
///         <td class="tp-int">u2</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i4</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">u2</td>
///         <td class="tp-int">u2</td>
///         <td class="tp-int">u4</td>
///         <td class="tp-int">u8</td>
///         <td class="tp-int">u16</td>
///         <td class="tp-int">u32</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">u4</th>
///         <td class="tp-int">u4</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i8</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">u4</td>
///         <td class="tp-int">u4</td>
///         <td class="tp-int">u4</td>
///         <td class="tp-int">u8</td>
///         <td class="tp-int">u16</td>
///         <td class="tp-int">u32</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">u8</th>
///         <td class="tp-int">u8</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">u8</td>
///         <td class="tp-int">u8</td>
///         <td class="tp-int">u8</td>
///         <td class="tp-int">u8</td>
///         <td class="tp-int">u16</td>
///         <td class="tp-int">u32</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">u16</th>
///         <td class="tp-int">u16</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">u16</td>
///         <td class="tp-int">u16</td>
///         <td class="tp-int">u16</td>
///         <td class="tp-int">u16</td>
///         <td class="tp-int">u16</td>
///         <td class="tp-int">u32</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">u32</th>
///         <td class="tp-int">u32</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">i64</td>
///         <td class="tp-int">u32</td>
///         <td class="tp-int">u32</td>
///         <td class="tp-int">u32</td>
///         <td class="tp-int">u32</td>
///         <td class="tp-int">u32</td>
///         <td class="tp-int">u32</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-int" scope="row">u64</th>
///         <td class="tp-int">u64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-int">u64</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-float" scope="row">bf16</th>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">bf16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-float" scope="row">f16</th>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f16</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-float" scope="row">f32</th>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f32</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-float" scope="row">f64</th>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-float">f64</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-complex" scope="row">c64</th>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c64</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///       <tr>
///         <th class="tp-complex" scope="row">c128</th>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///         <td class="tp-complex">c128</td>
///       </tr>
///     </tbody>
///   </table>
/// </div>
///
/// Mixing any integer type with a standard floating-point or complex type yields the smallest result type on that
/// branch that can represent both operands. For example, `i2 + f16 -> f16`, `u4 + f32 -> f32`, and `i64 + u64 -> f64`.
/// Low-precision custom floating-point types remain disconnected from the standard floating-point and complex part of
/// the lattice. However, one of them may still become the final result when it is already present in the expression
/// alongside another compatible custom floating-point type from the same disconnected group.
///
/// The type promotion logic is implemented in [`DataType::is_promotable_to`]. Note that these type promotion rules only
/// apply for automatic promotions. If you want to convert between [`DataType`]s violating these rules, you can still
/// do so explicitly using casting. `ryft` requires you to be explicit in such cases due to the risks around loss of
/// precision in arbitrary data type conversions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum DataType {
    /// [`DataType`] that represents token values that are threaded between side-effecting operations.
    /// This type is only used for values that contain a single token (i.e., that represent scalar values).
    Token,

    /// Boolean [`DataType`] that represents `true`/`false` values and can be promoted to numeric [`DataType`]s whose
    /// value sets can represent both boolean values.
    Boolean,

    /// [`DataType`] that represents 1-bit signed integer values, with the only representable values being `0` and `-1`.
    I1,

    /// [`DataType`] that represents 2-bit signed integer values, where the first bit corresponds to the sign.
    I2,

    /// [`DataType`] that represents 4-bit signed integer values, where the first bit corresponds to the sign.
    I4,

    /// [`DataType`] that represents 8-bit signed integer values, where the first bit corresponds to the sign.
    I8,

    /// [`DataType`] that represents 16-bit signed integer values, where the first bit corresponds to the sign.
    I16,

    /// [`DataType`] that represents 32-bit signed integer values, where the first bit corresponds to the sign.
    I32,

    /// [`DataType`] that represents 64-bit signed integer values, where the first bit corresponds to the sign.
    I64,

    /// [`DataType`] that represents 1-bit unsigned integer values.
    U1,

    /// [`DataType`] that represents 2-bit unsigned integer values.
    U2,

    /// [`DataType`] that represents 4-bit unsigned integer values.
    U4,

    /// [`DataType`] that represents 8-bit unsigned integer values.
    U8,

    /// [`DataType`] that represents 16-bit unsigned integer values.
    U16,

    /// [`DataType`] that represents 32-bit unsigned integer values.
    U32,

    /// [`DataType`] that represents 64-bit unsigned integer values.
    U64,

    /// [`DataType`] that represents 4-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E2M1 (1 bit for the sign, 2 bits for the exponent, and 1 bits for the mantissa).
    ///   - Exponent Bias: 1.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Not supported.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 1) * (1 * [E == 0] + M_0 * 2^-1)`.
    ///
    /// The `FN` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type can
    /// only represent finite values.
    F4E2M1FN,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E3M4 (1 bit for the sign, 3 bits for the exponent, and 4 bits for the mantissa).
    ///   - Exponent Bias: 3.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and arbitrary mantissa bit
    ///     values except for all `0`s.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 3) * (1 * [E == 0] + M_3 * 2^-1 + M_2 * 2^-2 + M_1 * 2^-3 + M_0 * 2^-4)`.
    F8E3M4,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E4M3 (1 bit for the sign, 4 bits for the exponent, and 3 bits for the mantissa).
    ///   - Exponent Bias: 7.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and arbitrary mantissa bit
    ///     values except for all `0`s.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 7) * (1 * [E == 0] + M_2 * 2^-1 + M_1 * 2^-2 + M_0 * 2^-3)`.
    F8E4M3,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E4M3 (1 bit for the sign, 4 bits for the exponent, and 3 bits for the mantissa).
    ///   - Exponent Bias: 7.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Supported with the sign bit set to `0` and all other bits set to `1`.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 7) * (1 * [E == 0] + M_2 * 2^-1 + M_1 * 2^-2 + M_0 * 2^-3)`.
    ///
    /// The `FN` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type can
    /// only represent finite values.
    F8E4M3FN,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E4M3 (1 bit for the sign, 4 bits for the exponent, and 3 bits for the mantissa).
    ///   - Exponent Bias: 8.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Supported with the sign bit set to `1` and all other bits set to `0`.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 8) * (1 * [E == 0] + M_2 * 2^-1 + M_1 * 2^-2 + M_0 * 2^-3)`.
    ///
    /// The `FNUZ` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type is
    /// not consistent with the IEEE 754 standard. The `FN` indicates that it can only represent finite values, and the
    /// `UZ` stands for "unsigned zero".
    F8E4M3FNUZ,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E4M3 (1 bit for the sign, 4 bits for the exponent, and 3 bits for the mantissa).
    ///   - Exponent Bias: 11.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Supported with the sign bit set to `1` and all other bits set to `0`.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 11) * (1 * [E == 0] + M_2 * 2^-1 + M_1 * 2^-2 + M_0 * 2^-3)`.
    ///
    /// The `FNUZ` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type is
    /// not consistent with the IEEE 754 standard. The `FN` indicates that it can only represent finite values, and the
    /// `UZ` stands for "unsigned zero".
    F8E4M3B11FNUZ,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E5M2 (1 bit for the sign, 5 bits for the exponent, and 2 bits for the mantissa).
    ///   - Exponent Bias: 15.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and the mantissa bits
    ///     set to `01`, `10`, or `11`.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 15) * (1 * [E == 0] + M_1 * 2^-1 + M_0 * 2^-2)`.
    F8E5M2,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E5M2 (1 bit for the sign, 5 bits for the exponent, and 2 bits for the mantissa).
    ///   - Exponent Bias: 16.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Supported with the sign bit set to `1` and all other bits set to `0`.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 16) * (1 * [E == 0] + M_2 * 2^-1 + M_1 * 2^-2 + M_0 * 2^-3)`.
    ///
    /// The `FNUZ` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type is
    /// not consistent with the IEEE 754 standard. The `FN` indicates that it can only represent finite values, and the
    /// `UZ` stands for "unsigned zero".
    F8E5M2FNUZ,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S0E8M0 (0 bits for the sign, 8 bits for the exponent, and 0 bits for the mantissa).
    ///   - Exponent Bias: 127.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Not supported.
    ///   - Denormalized: Not supported.
    ///
    /// The value of a number in this representation can be computed as:
    /// `2^(E - 127)`.
    ///
    /// The `FNU` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type can
    /// only represent finite values and that it has no sign bit.
    F8E8M0FNU,

    /// [`DataType`] that represents 16-bit floating-point values. This is not a
    /// standard type as defined by the IEEE 754 standard. Instead, it follows the
    /// [bfloat16 format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format),
    /// with the following characteristics:
    ///
    ///   - Bit Encoding: S1E8M7 (1 bit for the sign, 8 bits for the exponent, and 7 bits for the mantissa).
    ///   - Exponent Bias: 127.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and arbitrary mantissa bit
    ///     values except for all `0`s.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 127) * (1 * [E == 0] + M_6 * 2^-1 + ... + M_0 * 2^-7)`.
    ///
    /// [`DataType::BF16`] has lower precision but higher dynamic range compared to [`DataType::F16`].
    ///
    /// Refer to [this page](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) for more information
    /// on this [DataType].
    BF16,

    /// [`DataType`] that represents 16-bit floating-point values following the
    /// [IEEE 754 standard](https://en.wikipedia.org/wiki/IEEE_754), with the following characteristics:
    ///
    ///   - Bit Encoding: S1E5M10 (1 bit for the sign, 5 bits for the exponent, and 10 bits for the mantissa).
    ///   - Exponent Bias: 15.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and arbitrary mantissa bit
    ///     values except for all `0`s.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 15) * (1 * [E == 0] + M_9 * 2^-1 + ... + M_0 * 2^-10)`.
    ///
    /// Refer to [this page](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) for more information
    /// on this [DataType].
    F16,

    /// [`DataType`] that represents 32-bit floating-point values following the
    /// [IEEE 754 standard](https://en.wikipedia.org/wiki/IEEE_754), with the following characteristics:
    ///
    ///   - Bit Encoding: S1E8M23 (1 bit for the sign, 8 bits for the exponent, and 23 bits for the mantissa).
    ///   - Exponent Bias: 127.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and arbitrary mantissa bit
    ///     values except for all `0`s.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 127) * (1 * [E == 0] + M_22 * 2^-1 + ... + M_0 * 2^-23)`.
    ///
    /// Refer to [this page](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) for more information
    /// on this [DataType].
    F32,

    /// [`DataType`] that represents 64-bit floating-point values following the
    /// [IEEE 754 standard](https://en.wikipedia.org/wiki/IEEE_754), with the following characteristics:
    ///
    ///   - Bit Encoding: S1E11M52 (1 bit for the sign, 11 bits for the exponent, and 52 bits for the mantissa).
    ///   - Exponent Bias: 1023.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and arbitrary mantissa bit
    ///     values except for all `0`s.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 127) * (1 * [E == 0] + M_22 * 2^-1 + ... + M_0 * 2^-23)`.
    ///
    /// Refer to [this page](https://en.wikipedia.org/wiki/Double-precision_floating-point_format) for more information
    /// on this [DataType].
    F64,

    /// [`DataType`] that represents 64-bit complex numbers where the real and imaginary parts
    /// are [`DataType::F32`].
    C64,

    /// [`DataType`] that represents 128-bit complex numbers where the real and imaginary parts
    /// are [`DataType::F64`].
    C128,
}

impl DataType {
    /// All [`DataType`] values.
    const ALL: [Self; 31] = [
        Self::Token,
        Self::Boolean,
        Self::I1,
        Self::I2,
        Self::I4,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::U1,
        Self::U2,
        Self::U4,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::F4E2M1FN,
        Self::F8E3M4,
        Self::F8E4M3,
        Self::F8E4M3FN,
        Self::F8E4M3FNUZ,
        Self::F8E4M3B11FNUZ,
        Self::F8E5M2,
        Self::F8E5M2FNUZ,
        Self::F8E8M0FNU,
        Self::BF16,
        Self::F16,
        Self::F32,
        Self::F64,
        Self::C64,
        Self::C128,
    ];

    /// Number of [`DataType`] values.
    const COUNT: usize = Self::ALL.len();

    /// Returns the stable index of this [`DataType`] in [`DataType::ALL`].
    const fn index(self) -> usize {
        match self {
            Self::Token => 0,
            Self::Boolean => 1,
            Self::I1 => 2,
            Self::I2 => 3,
            Self::I4 => 4,
            Self::I8 => 5,
            Self::I16 => 6,
            Self::I32 => 7,
            Self::I64 => 8,
            Self::U1 => 9,
            Self::U2 => 10,
            Self::U4 => 11,
            Self::U8 => 12,
            Self::U16 => 13,
            Self::U32 => 14,
            Self::U64 => 15,
            Self::F4E2M1FN => 16,
            Self::F8E3M4 => 17,
            Self::F8E4M3 => 18,
            Self::F8E4M3FN => 19,
            Self::F8E4M3FNUZ => 20,
            Self::F8E4M3B11FNUZ => 21,
            Self::F8E5M2 => 22,
            Self::F8E5M2FNUZ => 23,
            Self::F8E8M0FNU => 24,
            Self::BF16 => 25,
            Self::F16 => 26,
            Self::F32 => 27,
            Self::F64 => 28,
            Self::C64 => 29,
            Self::C128 => 30,
        }
    }
}

/// Node in the [`DataType`] promotion lattice. Refer to the documentation of [`DataType`] for more information
/// on `ryft`'s type promotion semantics.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum DataTypePromotionNode {
    /// Represents a concrete [`DataType`] node in the promotion lattice.
    DataType(DataType),

    /// Represents a [`DataType`] in the promotion lattice that can be promoted to any floating-point type,
    /// depending on how it interacts with other types.
    FloatingPointDataType,
}

impl DataTypePromotionNode {
    /// All [`DataTypePromotionNode`] values.
    const ALL: [Self; DataType::COUNT + 1] = [
        Self::DataType(DataType::Token),
        Self::DataType(DataType::Boolean),
        Self::DataType(DataType::I1),
        Self::DataType(DataType::I2),
        Self::DataType(DataType::I4),
        Self::DataType(DataType::I8),
        Self::DataType(DataType::I16),
        Self::DataType(DataType::I32),
        Self::DataType(DataType::I64),
        Self::DataType(DataType::U1),
        Self::DataType(DataType::U2),
        Self::DataType(DataType::U4),
        Self::DataType(DataType::U8),
        Self::DataType(DataType::U16),
        Self::DataType(DataType::U32),
        Self::DataType(DataType::U64),
        Self::DataType(DataType::F4E2M1FN),
        Self::DataType(DataType::F8E3M4),
        Self::DataType(DataType::F8E4M3),
        Self::DataType(DataType::F8E4M3FN),
        Self::DataType(DataType::F8E4M3FNUZ),
        Self::DataType(DataType::F8E4M3B11FNUZ),
        Self::DataType(DataType::F8E5M2),
        Self::DataType(DataType::F8E5M2FNUZ),
        Self::DataType(DataType::F8E8M0FNU),
        Self::DataType(DataType::BF16),
        Self::DataType(DataType::F16),
        Self::DataType(DataType::F32),
        Self::DataType(DataType::F64),
        Self::DataType(DataType::C64),
        Self::DataType(DataType::C128),
        Self::FloatingPointDataType,
    ];

    /// Number of [`DataTypePromotionNode`] values.
    const COUNT: usize = Self::ALL.len();

    /// Returns the stable index of this [`DataTypePromotionNode`] in [`DataTypePromotionNode::ALL`].
    const fn index(self) -> usize {
        match self {
            Self::DataType(data_type) => data_type.index(),
            Self::FloatingPointDataType => DataType::COUNT,
        }
    }

    /// Returns a one-hot bit encoding that represents this [`DataTypePromotionNode`]. All bits of the returned `u64`
    /// are set to `0` except for the [`DataTypePromotionNode::index`]-th bit which is set to 1.
    const fn bitmask(self) -> u64 {
        1u64 << self.index()
    }
}

/// Transitive [`DataType`] promotion upper-bound bitmasks for each [`DataTypePromotionNode`]. The resulting array
/// contains one bitmask for each [`DataTypePromotionNode`] represented as a `u64`. The bits of that bitmask that are
/// set to `1` correspond to the [`DataTypePromotionNode`]s that that [`DataTypePromotionNode`] is can be promoted to.
static DATA_TYPE_PROMOTION_UPPER_BOUNDS_BITMASKS: [u64; DataTypePromotionNode::COUNT] = {
    let mut bitmasks = [0_u64; DataTypePromotionNode::COUNT];
    let mut index = 0;
    while index < DataTypePromotionNode::COUNT {
        let node = DataTypePromotionNode::ALL[index];
        bitmasks[index] = node.bitmask()
            | match node {
                DataTypePromotionNode::DataType(DataType::Token) => 0,
                DataTypePromotionNode::DataType(DataType::Boolean) => {
                    DataTypePromotionNode::DataType(DataType::U1).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::I1) => {
                    DataTypePromotionNode::DataType(DataType::I2).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::I2) => {
                    DataTypePromotionNode::DataType(DataType::I4).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::I4) => {
                    DataTypePromotionNode::DataType(DataType::I8).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::I8) => {
                    DataTypePromotionNode::DataType(DataType::I16).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::I16) => {
                    DataTypePromotionNode::DataType(DataType::I32).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::I32) => {
                    DataTypePromotionNode::DataType(DataType::I64).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::I64) => {
                    DataTypePromotionNode::FloatingPointDataType.bitmask()
                }
                DataTypePromotionNode::DataType(DataType::U1) => {
                    DataTypePromotionNode::DataType(DataType::I2).bitmask()
                        | DataTypePromotionNode::DataType(DataType::U2).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::U2) => {
                    DataTypePromotionNode::DataType(DataType::I4).bitmask()
                        | DataTypePromotionNode::DataType(DataType::U4).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::U4) => {
                    DataTypePromotionNode::DataType(DataType::I8).bitmask()
                        | DataTypePromotionNode::DataType(DataType::U8).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::U8) => {
                    DataTypePromotionNode::DataType(DataType::I16).bitmask()
                        | DataTypePromotionNode::DataType(DataType::U16).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::U16) => {
                    DataTypePromotionNode::DataType(DataType::I32).bitmask()
                        | DataTypePromotionNode::DataType(DataType::U32).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::U32) => {
                    DataTypePromotionNode::DataType(DataType::I64).bitmask()
                        | DataTypePromotionNode::DataType(DataType::U64).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::U64) => {
                    DataTypePromotionNode::FloatingPointDataType.bitmask()
                }
                DataTypePromotionNode::DataType(DataType::F4E2M1FN) => 0,
                DataTypePromotionNode::DataType(DataType::F8E3M4) => 0,
                DataTypePromotionNode::DataType(DataType::F8E4M3) => 0,
                DataTypePromotionNode::DataType(DataType::F8E4M3FN) => 0,
                DataTypePromotionNode::DataType(DataType::F8E4M3FNUZ) => 0,
                DataTypePromotionNode::DataType(DataType::F8E4M3B11FNUZ) => 0,
                DataTypePromotionNode::DataType(DataType::F8E5M2) => 0,
                DataTypePromotionNode::DataType(DataType::F8E5M2FNUZ) => 0,
                DataTypePromotionNode::DataType(DataType::F8E8M0FNU) => 0,
                DataTypePromotionNode::DataType(DataType::BF16) => {
                    DataTypePromotionNode::DataType(DataType::F32).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::F16) => {
                    DataTypePromotionNode::DataType(DataType::F32).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::F32) => {
                    DataTypePromotionNode::DataType(DataType::F64).bitmask()
                        | DataTypePromotionNode::DataType(DataType::C64).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::F64) => {
                    DataTypePromotionNode::DataType(DataType::C128).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::C64) => {
                    DataTypePromotionNode::DataType(DataType::C128).bitmask()
                }
                DataTypePromotionNode::DataType(DataType::C128) => 0,
                DataTypePromotionNode::FloatingPointDataType => {
                    DataTypePromotionNode::DataType(DataType::BF16).bitmask()
                        | DataTypePromotionNode::DataType(DataType::F16).bitmask()
                        | DataTypePromotionNode::DataType(DataType::C64).bitmask()
                }
            };
        index += 1;
    }

    // Compute the transitive closure by allowing each node to act as an intermediate promotion step.
    let mut pivot_index = 0;
    while pivot_index < DataTypePromotionNode::COUNT {
        let pivot_bitmask = bitmasks[pivot_index];
        let pivot_bit = 1_u64 << pivot_index;
        let mut lhs_index = 0;
        while lhs_index < DataTypePromotionNode::COUNT {
            if bitmasks[lhs_index] & pivot_bit != 0 {
                bitmasks[lhs_index] |= pivot_bitmask;
            }
            lhs_index += 1;
        }
        pivot_index += 1;
    }
    bitmasks
};

/// Least-upper-bound lookup table for concrete [`DataType`] promotion pairs.
/// `DATA_TYPE_PROMOTION_LEAST_UPPER_BOUNDS[x][y]` stores the unique promoted [`DataType`] for the ordered pair
/// `(x, y)`. An entry is `None` when the two types have no automatic promotion path or when the promotion lattice
/// would yield an ambiguous result.
static DATA_TYPE_PROMOTION_LEAST_UPPER_BOUNDS: [[Option<DataType>; DataType::COUNT]; DataType::COUNT] = {
    let mut least_upper_bounds = [[None; DataType::COUNT]; DataType::COUNT];
    let mut lhs_index = 0;
    while lhs_index < DataType::COUNT {
        let lhs = DataTypePromotionNode::ALL[lhs_index];
        let lhs_upper_bounds_bitmask = DATA_TYPE_PROMOTION_UPPER_BOUNDS_BITMASKS[lhs.index()];
        let mut rhs_index = lhs_index;
        while rhs_index < DataType::COUNT {
            let rhs = DataTypePromotionNode::ALL[rhs_index];
            let rhs_upper_bounds_bitmask = DATA_TYPE_PROMOTION_UPPER_BOUNDS_BITMASKS[rhs.index()];
            let common_upper_bounds = lhs_upper_bounds_bitmask & rhs_upper_bounds_bitmask;
            let mut least_upper_bound = None;
            let mut candidate_index = 0;
            while candidate_index < DataTypePromotionNode::COUNT {
                let candidate_bit = 1_u64 << candidate_index;
                let candidate_upper_bounds_bitmask = DATA_TYPE_PROMOTION_UPPER_BOUNDS_BITMASKS[candidate_index];
                if common_upper_bounds & candidate_bit != 0
                    && candidate_upper_bounds_bitmask & common_upper_bounds == common_upper_bounds
                {
                    let candidate = match DataTypePromotionNode::ALL[candidate_index] {
                        DataTypePromotionNode::DataType(data_type) => data_type,
                        DataTypePromotionNode::FloatingPointDataType => DataType::F64,
                    };
                    match least_upper_bound {
                        None => least_upper_bound = Some(candidate),
                        Some(existing) if existing.index() == candidate.index() => {}
                        Some(_) => {
                            least_upper_bound = None;
                            break;
                        }
                    }
                }
                candidate_index += 1;
            }
            least_upper_bounds[lhs_index][rhs_index] = least_upper_bound;
            least_upper_bounds[rhs_index][lhs_index] = least_upper_bound;
            rhs_index += 1;
        }
        lhs_index += 1;
    }
    least_upper_bounds
};

impl DataType {
    /// Returns the promoted [`DataType`] for the provided data types, which is defined as the least upper bound of
    /// the input [`DataType`]s in the type promotion lattice described in the documentation of [`DataType`]. In other
    /// words, it returns the _smallest_ [`DataType`] that every input type can be promoted to, automatically. Note
    /// that the returned type is not required to be one of the inputs. For example, combining [`DataType::I64`] with
    /// [`DataType::U64`] yields [`DataType::F64`] because there is no wider integer type that can represent both.
    /// This operation is order-invariant, meaning that it returns the same result irrespective of the order in which
    /// the input data types are provided.
    ///
    /// This function returns [`DataTypeError::EmptyDataTypePromotionInput`] when `data_types` is empty and
    /// [`DataTypeError::InvalidDataTypePromotion`] when the provided types are incompatible in terms of type
    /// promotion.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::DataType;
    /// assert_eq!(DataType::promoted(&[DataType::Boolean]), Ok(DataType::Boolean));
    /// assert_eq!(DataType::promoted(&[DataType::Boolean, DataType::U16]), Ok(DataType::U16));
    /// assert_eq!(DataType::promoted(&[DataType::Boolean, DataType::F32]), Ok(DataType::F32));
    /// assert_eq!(DataType::promoted(&[DataType::F32, DataType::U16]), Ok(DataType::F32));
    /// assert_eq!(DataType::promoted(&[DataType::Boolean, DataType::U16, DataType::F32]), Ok(DataType::F32));
    /// assert_eq!(DataType::promoted(&[DataType::I64, DataType::U64]), Ok(DataType::F64));
    /// ```
    #[inline]
    pub fn promoted(data_types: &[Self]) -> Result<Self, DataTypeError> {
        let Some((head, tail)) = data_types.split_first() else {
            return Err(DataTypeError::empty_data_type_promotion_input());
        };
        tail.iter().try_fold(*head, |x, y| {
            DATA_TYPE_PROMOTION_LEAST_UPPER_BOUNDS[x.index()][y.index()].ok_or_else(|| {
                DataTypeError::invalid_data_type_promotion(format!(
                    "cannot promote types `{x}` and `{y}` to a common type",
                ))
            })
        })
    }

    /// Promotes this [`DataType`] to the provided [`DataType`]. This function returns
    /// [`DataTypeError::InvalidDataTypePromotion`] when the promotion is not allowed. Refer to the documentation of
    /// [`DataType`] for more information on type promotions and the rules that govern them.
    #[inline]
    pub fn promote_to(self, other: DataType) -> Result<DataType, DataTypeError> {
        if self.is_promotable_to(other) {
            Ok(other)
        } else {
            Err(DataTypeError::invalid_data_type_promotion(format!("cannot promote type `{self}` to type `{other}`")))
        }
    }

    /// Returns `true` if this [`DataType`] can be promoted to the provided [`DataType`]. Note that this function will
    /// always return `true` when `self == other`. Refer to the documentation of [`DataType`] for more information on
    /// type promotions and the rules that govern them.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::DataType;
    /// assert!(DataType::I32.is_promotable_to(DataType::F64));
    /// assert!(DataType::F32.is_promotable_to(DataType::C64));
    /// assert!(!DataType::F64.is_promotable_to(DataType::I32));
    /// ```
    #[inline]
    pub fn is_promotable_to(self, other: Self) -> bool {
        DATA_TYPE_PROMOTION_UPPER_BOUNDS_BITMASKS[self.index()] & DataTypePromotionNode::DataType(other).bitmask() != 0
    }
}

impl Display for DataType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            DataType::Token => "token",
            DataType::Boolean => "bool",
            DataType::I1 => "i1",
            DataType::I2 => "i2",
            DataType::I4 => "i4",
            DataType::I8 => "i8",
            DataType::I16 => "i16",
            DataType::I32 => "i32",
            DataType::I64 => "i64",
            DataType::U1 => "u1",
            DataType::U2 => "u2",
            DataType::U4 => "u4",
            DataType::U8 => "u8",
            DataType::U16 => "u16",
            DataType::U32 => "u32",
            DataType::U64 => "u64",
            DataType::F4E2M1FN => "f4e2m1fn",
            DataType::F8E3M4 => "f8e3m4",
            DataType::F8E4M3 => "f8e4m3",
            DataType::F8E4M3FN => "f8e4m3fn",
            DataType::F8E4M3FNUZ => "f8e4m3fnuz",
            DataType::F8E4M3B11FNUZ => "f8e4m3b11fnuz",
            DataType::F8E5M2 => "f8e5m2",
            DataType::F8E5M2FNUZ => "f8e5m2fnuz",
            DataType::F8E8M0FNU => "f8e8m0fnu",
            DataType::BF16 => "bf16",
            DataType::F16 => "f16",
            DataType::F32 => "f32",
            DataType::F64 => "f64",
            DataType::C64 => "c64",
            DataType::C128 => "c128",
        })
    }
}

impl Type for DataType {
    #[inline]
    fn is_subtype_of(&self, other: &Self) -> bool {
        // Note that this is not quite a subtyping relationship in that certain type promotions can result in loss
        // of information for values of those types (e.g., [`DataType::U64`] to [`DataType::F64`]). However, this
        // is intended to make ergonomics better for when working with `ryft` for scientific applications.
        self.is_promotable_to(*other)
    }
}

#[cfg(feature = "xla")]
impl DataType {
    /// Creates a [`DataType`] from the provided PJRT [`BufferType`]. Returns [`DataTypeError::InvalidDataType`]
    /// when the provided [`BufferType`] is [`BufferType::Invalid``], which is a PJRT-internal sentinel value.
    pub fn from_pjrt_buffer_type(buffer_type: BufferType) -> Result<Self, DataTypeError> {
        buffer_type.try_into()
    }

    /// Returns the PJRT [`BufferType`] that corresponds to this [`DataType`].
    pub fn to_pjrt_buffer_type(self) -> BufferType {
        self.into()
    }
}

#[cfg(feature = "xla")]
impl TryFrom<BufferType> for DataType {
    type Error = DataTypeError;

    fn try_from(buffer_type: BufferType) -> Result<Self, Self::Error> {
        match buffer_type {
            BufferType::Invalid => {
                Err(DataTypeError::invalid_data_type(format!("invalid data type from PJRT: '{buffer_type}'")))
            }
            BufferType::Token => Ok(Self::Token),
            BufferType::Predicate => Ok(Self::Boolean),
            BufferType::I1 => Ok(Self::I1),
            BufferType::I2 => Ok(Self::I2),
            BufferType::I4 => Ok(Self::I4),
            BufferType::I8 => Ok(Self::I8),
            BufferType::I16 => Ok(Self::I16),
            BufferType::I32 => Ok(Self::I32),
            BufferType::I64 => Ok(Self::I64),
            BufferType::U1 => Ok(Self::U1),
            BufferType::U2 => Ok(Self::U2),
            BufferType::U4 => Ok(Self::U4),
            BufferType::U8 => Ok(Self::U8),
            BufferType::U16 => Ok(Self::U16),
            BufferType::U32 => Ok(Self::U32),
            BufferType::U64 => Ok(Self::U64),
            BufferType::F4E2M1FN => Ok(Self::F4E2M1FN),
            BufferType::F8E3M4 => Ok(Self::F8E3M4),
            BufferType::F8E4M3 => Ok(Self::F8E4M3),
            BufferType::F8E4M3FN => Ok(Self::F8E4M3FN),
            BufferType::F8E4M3FNUZ => Ok(Self::F8E4M3FNUZ),
            BufferType::F8E4M3B11FNUZ => Ok(Self::F8E4M3B11FNUZ),
            BufferType::F8E5M2 => Ok(Self::F8E5M2),
            BufferType::F8E5M2FNUZ => Ok(Self::F8E5M2FNUZ),
            BufferType::F8E8M0FNU => Ok(Self::F8E8M0FNU),
            BufferType::BF16 => Ok(Self::BF16),
            BufferType::F16 => Ok(Self::F16),
            BufferType::F32 => Ok(Self::F32),
            BufferType::F64 => Ok(Self::F64),
            BufferType::C64 => Ok(Self::C64),
            BufferType::C128 => Ok(Self::C128),
        }
    }
}

#[cfg(feature = "xla")]
impl From<DataType> for BufferType {
    fn from(data_type: DataType) -> Self {
        match data_type {
            DataType::Token => Self::Token,
            DataType::Boolean => Self::Predicate,
            DataType::I1 => Self::I1,
            DataType::I2 => Self::I2,
            DataType::I4 => Self::I4,
            DataType::I8 => Self::I8,
            DataType::I16 => Self::I16,
            DataType::I32 => Self::I32,
            DataType::I64 => Self::I64,
            DataType::U1 => Self::U1,
            DataType::U2 => Self::U2,
            DataType::U4 => Self::U4,
            DataType::U8 => Self::U8,
            DataType::U16 => Self::U16,
            DataType::U32 => Self::U32,
            DataType::U64 => Self::U64,
            DataType::F4E2M1FN => Self::F4E2M1FN,
            DataType::F8E3M4 => Self::F8E3M4,
            DataType::F8E4M3 => Self::F8E4M3,
            DataType::F8E4M3FN => Self::F8E4M3FN,
            DataType::F8E4M3FNUZ => Self::F8E4M3FNUZ,
            DataType::F8E4M3B11FNUZ => Self::F8E4M3B11FNUZ,
            DataType::F8E5M2 => Self::F8E5M2,
            DataType::F8E5M2FNUZ => Self::F8E5M2FNUZ,
            DataType::F8E8M0FNU => Self::F8E8M0FNU,
            DataType::BF16 => Self::BF16,
            DataType::F16 => Self::F16,
            DataType::F32 => Self::F32,
            DataType::F64 => Self::F64,
            DataType::C64 => Self::C64,
            DataType::C128 => Self::C128,
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "xla")]
    use super::BufferType;
    use super::{DataType, DataTypeError};

    #[test]
    fn test_data_type_promoted() {
        assert_eq!(DataType::promoted(&[DataType::Boolean]), Ok(DataType::Boolean));
        assert_eq!(DataType::promoted(&[DataType::I2]), Ok(DataType::I2));
        assert_eq!(DataType::promoted(&[DataType::F8E3M4]), Ok(DataType::F8E3M4));
        assert_eq!(DataType::promoted(&[DataType::C128]), Ok(DataType::C128));
        assert_eq!(
            DataType::promoted(&[DataType::Boolean, DataType::Boolean, DataType::Boolean]),
            Ok(DataType::Boolean),
        );
        assert_eq!(DataType::promoted(&[DataType::F8E3M4, DataType::F8E3M4, DataType::F8E3M4]), Ok(DataType::F8E3M4));
        assert_eq!(DataType::promoted(&[DataType::U16, DataType::U16, DataType::U16]), Ok(DataType::U16));
        assert_eq!(DataType::promoted(&[DataType::Boolean, DataType::U1]), Ok(DataType::U1));
        assert_eq!(DataType::promoted(&[DataType::U1, DataType::Boolean]), Ok(DataType::U1));
        assert_eq!(DataType::promoted(&[DataType::Boolean, DataType::I1]), Ok(DataType::I2));
        assert_eq!(DataType::promoted(&[DataType::I1, DataType::Boolean]), Ok(DataType::I2));
        assert_eq!(DataType::promoted(&[DataType::Boolean, DataType::C64]), Ok(DataType::C64));
        assert_eq!(DataType::promoted(&[DataType::I2, DataType::U2]), Ok(DataType::I4));
        assert_eq!(DataType::promoted(&[DataType::U2, DataType::I2]), Ok(DataType::I4));
        assert_eq!(DataType::promoted(&[DataType::I4, DataType::U4]), Ok(DataType::I8));
        assert_eq!(DataType::promoted(&[DataType::U4, DataType::I4]), Ok(DataType::I8));
        assert_eq!(DataType::promoted(&[DataType::U8, DataType::I8]), Ok(DataType::I16));
        assert_eq!(DataType::promoted(&[DataType::I8, DataType::U8]), Ok(DataType::I16));
        assert_eq!(DataType::promoted(&[DataType::U16, DataType::I16]), Ok(DataType::I32));
        assert_eq!(DataType::promoted(&[DataType::I16, DataType::U16]), Ok(DataType::I32));
        assert_eq!(DataType::promoted(&[DataType::U32, DataType::I32]), Ok(DataType::I64));
        assert_eq!(DataType::promoted(&[DataType::I32, DataType::U32]), Ok(DataType::I64));
        assert_eq!(DataType::promoted(&[DataType::U64, DataType::I64]), Ok(DataType::F64));
        assert_eq!(DataType::promoted(&[DataType::I64, DataType::U64]), Ok(DataType::F64));
        assert_eq!(DataType::promoted(&[DataType::F16, DataType::BF16]), Ok(DataType::F32));
        assert_eq!(DataType::promoted(&[DataType::BF16, DataType::F16]), Ok(DataType::F32));
        assert_eq!(DataType::promoted(&[DataType::Boolean, DataType::U1, DataType::I1]), Ok(DataType::I2));
        assert_eq!(DataType::promoted(&[DataType::I1, DataType::Boolean, DataType::U1]), Ok(DataType::I2));
        assert_eq!(DataType::promoted(&[DataType::U1, DataType::I1, DataType::Boolean]), Ok(DataType::I2));
        assert_eq!(DataType::promoted(&[DataType::Boolean, DataType::U16, DataType::C64]), Ok(DataType::C64));
        assert_eq!(DataType::promoted(&[DataType::C64, DataType::Boolean, DataType::U16]), Ok(DataType::C64));
        assert_eq!(DataType::promoted(&[DataType::U16, DataType::C64, DataType::Boolean]), Ok(DataType::C64));
        assert_eq!(DataType::promoted(&[DataType::Boolean, DataType::U16, DataType::F32]), Ok(DataType::F32));
        assert_eq!(DataType::promoted(&[DataType::F32, DataType::Boolean, DataType::U16]), Ok(DataType::F32));
        assert_eq!(DataType::promoted(&[DataType::U16, DataType::F32, DataType::Boolean]), Ok(DataType::F32));
        assert_eq!(DataType::promoted(&[DataType::F32, DataType::I8, DataType::BF16]), Ok(DataType::F32));
        assert_eq!(DataType::promoted(&[DataType::BF16, DataType::F32, DataType::I8]), Ok(DataType::F32));
        assert_eq!(DataType::promoted(&[DataType::I8, DataType::BF16, DataType::F32]), Ok(DataType::F32));
        assert_eq!(DataType::promoted(&[DataType::F16, DataType::BF16, DataType::F64]), Ok(DataType::F64));
        assert_eq!(DataType::promoted(&[DataType::F64, DataType::F16, DataType::BF16]), Ok(DataType::F64));
        assert_eq!(DataType::promoted(&[DataType::BF16, DataType::F64, DataType::F16]), Ok(DataType::F64));
        assert!(matches!(DataType::promoted(&[]), Err(DataTypeError::EmptyDataTypePromotionInput { .. }),));
        assert!(matches!(
            DataType::promoted(&[DataType::F8E3M4, DataType::F32]),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote types `f8e3m4` and `f32` to a common type",
        ));
        assert!(matches!(
            DataType::promoted(&[DataType::F32, DataType::F8E3M4]),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote types `f32` and `f8e3m4` to a common type",
        ));
        assert!(matches!(
            DataType::promoted(&[DataType::U4, DataType::F8E3M4]),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote types `u4` and `f8e3m4` to a common type",
        ));
        assert!(matches!(
            DataType::promoted(&[DataType::Boolean, DataType::U1, DataType::F8E3M4]),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote types `u1` and `f8e3m4` to a common type",
        ));
        assert!(matches!(
            DataType::promoted(&[DataType::F8E3M4, DataType::F8E8M0FNU]),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote types `f8e3m4` and `f8e8m0fnu` to a common type",
        ));
        assert!(matches!(
            DataType::promoted(&[DataType::F8E8M0FNU, DataType::F8E3M4]),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote types `f8e8m0fnu` and `f8e3m4` to a common type",
        ));
        assert!(matches!(
            DataType::promoted(&[DataType::F4E2M1FN, DataType::F8E3M4]),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote types `f4e2m1fn` and `f8e3m4` to a common type",
        ));
    }

    #[test]
    fn test_data_type_promote_to() {
        assert_eq!(DataType::Boolean.promote_to(DataType::Boolean), Ok(DataType::Boolean));
        assert_eq!(DataType::Boolean.promote_to(DataType::U1), Ok(DataType::U1));
        assert_eq!(DataType::I1.promote_to(DataType::I2), Ok(DataType::I2));
        assert_eq!(DataType::I2.promote_to(DataType::I4), Ok(DataType::I4));
        assert_eq!(DataType::U1.promote_to(DataType::I2), Ok(DataType::I2));
        assert_eq!(DataType::U1.promote_to(DataType::U2), Ok(DataType::U2));
        assert_eq!(DataType::U2.promote_to(DataType::I4), Ok(DataType::I4));
        assert_eq!(DataType::U4.promote_to(DataType::I8), Ok(DataType::I8));
        assert_eq!(DataType::I2.promote_to(DataType::F32), Ok(DataType::F32));
        assert_eq!(DataType::U8.promote_to(DataType::I16), Ok(DataType::I16));
        assert_eq!(DataType::BF16.promote_to(DataType::F32), Ok(DataType::F32));
        assert_eq!(DataType::F64.promote_to(DataType::C128), Ok(DataType::C128));
        assert!(matches!(
            DataType::Boolean.promote_to(DataType::I1),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote type `bool` to type `i1`",
        ));
        assert!(matches!(
            DataType::U4.promote_to(DataType::I4),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote type `u4` to type `i4`",
        ));
        assert!(matches!(
            DataType::I16.promote_to(DataType::Boolean),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote type `i16` to type `bool`",
        ));
        assert!(matches!(
            DataType::F8E5M2FNUZ.promote_to(DataType::BF16),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote type `f8e5m2fnuz` to type `bf16`",
        ));
        assert!(matches!(
            DataType::F8E3M4.promote_to(DataType::F4E2M1FN),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote type `f8e3m4` to type `f4e2m1fn`",
        ));
        assert!(matches!(
            DataType::F8E3M4.promote_to(DataType::F8E4M3FN),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote type `f8e3m4` to type `f8e4m3fn`",
        ));
        assert!(matches!(
            DataType::F64.promote_to(DataType::C64),
            Err(DataTypeError::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote type `f64` to type `c64`",
        ));
    }

    #[test]
    fn test_data_type_promotable_to() {
        assert!(DataType::Boolean.is_promotable_to(DataType::U1));
        assert!(DataType::Boolean.is_promotable_to(DataType::BF16));
        assert!(DataType::Boolean.is_promotable_to(DataType::C128));
        assert!(DataType::I1.is_promotable_to(DataType::I2));
        assert!(DataType::I2.is_promotable_to(DataType::I4));
        assert!(DataType::I2.is_promotable_to(DataType::F32));
        assert!(DataType::U1.is_promotable_to(DataType::I2));
        assert!(DataType::U1.is_promotable_to(DataType::U2));
        assert!(DataType::U2.is_promotable_to(DataType::I4));
        assert!(DataType::U4.is_promotable_to(DataType::I8));
        assert!(DataType::U8.is_promotable_to(DataType::I16));
        assert!(DataType::U16.is_promotable_to(DataType::I32));
        assert!(DataType::F4E2M1FN.is_promotable_to(DataType::F4E2M1FN));
        assert!(DataType::F16.is_promotable_to(DataType::F32));
        assert!(DataType::F32.is_promotable_to(DataType::C64));
        assert!(!DataType::Boolean.is_promotable_to(DataType::I1));
        assert!(!DataType::U8.is_promotable_to(DataType::I8));
        assert!(!DataType::U4.is_promotable_to(DataType::I4));
        assert!(!DataType::U4.is_promotable_to(DataType::F8E3M4));
        assert!(!DataType::I64.is_promotable_to(DataType::F8E3M4));
        assert!(!DataType::F4E2M1FN.is_promotable_to(DataType::BF16));
        assert!(!DataType::F8E3M4.is_promotable_to(DataType::F8E4M3FN));
        assert!(!DataType::F8E4M3B11FNUZ.is_promotable_to(DataType::BF16));
    }

    #[test]
    fn test_data_type_to_string() {
        assert_eq!(DataType::Token.to_string(), "token");
        assert_eq!(DataType::Boolean.to_string(), "bool");
        assert_eq!(DataType::U4.to_string(), "u4");
        assert_eq!(DataType::I64.to_string(), "i64");
        assert_eq!(DataType::F8E3M4.to_string(), "f8e3m4");
        assert_eq!(DataType::F8E4M3FNUZ.to_string(), "f8e4m3fnuz");
        assert_eq!(DataType::BF16.to_string(), "bf16");
        assert_eq!(DataType::F64.to_string(), "f64");
        assert_eq!(DataType::C128.to_string(), "c128");
    }

    #[cfg(feature = "xla")]
    #[test]
    fn test_data_type_from_and_to_pjrt_buffer_type() {
        assert!(matches!(
            DataType::from_pjrt_buffer_type(BufferType::Invalid),
            Err(DataTypeError::InvalidDataType { message, .. }) if message == "invalid data type from PJRT: 'invalid'",
        ));
        for &(data_type, buffer_type) in &[
            (DataType::Token, BufferType::Token),
            (DataType::Boolean, BufferType::Predicate),
            (DataType::I1, BufferType::I1),
            (DataType::I2, BufferType::I2),
            (DataType::I4, BufferType::I4),
            (DataType::I8, BufferType::I8),
            (DataType::I16, BufferType::I16),
            (DataType::I32, BufferType::I32),
            (DataType::I64, BufferType::I64),
            (DataType::U1, BufferType::U1),
            (DataType::U2, BufferType::U2),
            (DataType::U4, BufferType::U4),
            (DataType::U8, BufferType::U8),
            (DataType::U16, BufferType::U16),
            (DataType::U32, BufferType::U32),
            (DataType::U64, BufferType::U64),
            (DataType::F4E2M1FN, BufferType::F4E2M1FN),
            (DataType::F8E3M4, BufferType::F8E3M4),
            (DataType::F8E4M3, BufferType::F8E4M3),
            (DataType::F8E4M3FN, BufferType::F8E4M3FN),
            (DataType::F8E4M3FNUZ, BufferType::F8E4M3FNUZ),
            (DataType::F8E4M3B11FNUZ, BufferType::F8E4M3B11FNUZ),
            (DataType::F8E5M2, BufferType::F8E5M2),
            (DataType::F8E5M2FNUZ, BufferType::F8E5M2FNUZ),
            (DataType::F8E8M0FNU, BufferType::F8E8M0FNU),
            (DataType::BF16, BufferType::BF16),
            (DataType::F16, BufferType::F16),
            (DataType::F32, BufferType::F32),
            (DataType::F64, BufferType::F64),
            (DataType::C64, BufferType::C64),
            (DataType::C128, BufferType::C128),
        ] {
            assert_eq!(DataType::from_pjrt_buffer_type(buffer_type), Ok(data_type));
            assert_eq!(data_type.to_pjrt_buffer_type(), buffer_type);
        }
    }
}
