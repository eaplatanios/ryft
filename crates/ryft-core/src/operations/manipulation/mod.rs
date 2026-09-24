//! Operations that rearrange, select, and relocate array data without changing element values. Each operation is
//! defined by an [`Operation`](crate::Operation) type (e.g., [`SliceOperation`]) together with a value capability
//! trait (e.g., [`Slice`]) whose functions apply it to eager [`Array`](crate::Array)s and traced values alike, so
//! the same code executes immediately or records into a program depending on the value it runs on.
//!
//! The operations fall into four groups:
//!
//!   - **Reordering without Changing the Shape:** [`Transpose`] permutes axes and [`Reverse`] flips the order of the
//!     elements along selected axes.
//!   - **Changing the shape:** [`Reshape`] reinterprets the element layout, [`Broadcast`] repeats an array along new
//!     or size-one axes, [`Concatenate`] joins arrays along an axis, and [`Pad`] adds, removes, or interleaves edge
//!     and interior padding.
//!   - **Selecting and Updating Elements:** [`Slice`] and [`UpdateSlice`] take static windows, [`DynamicSlice`] and
//!     [`DynamicUpdateSlice`] take runtime start indices, [`DynamicSliceWithDimensions`] takes dimension-valued
//!     windows, [`Gather`] and [`Scatter`] read and write arbitrary coordinates, and [`Indexing`] composes all of
//!     these behind an [`index!`](crate::index) selector list with
//!     [NumPy-style](https://numpy.org/doc/stable/user/basics.indexing.html) integers, ranges, strides, new axes,
//!     array indices, and masks. On array values the selection reads or returns updated copies, and on reference
//!     values it derives a view that is read and written in place.
//!   - **Changing Representation and Placement:** [`ConvertElementType`] changes the element data type and
//!     [`TransferToMemory`] moves an array between memory spaces.
//!
//! Negative host integers and negative runtime start indices count from the end of their axis once before clamping.
//! The `Dynamic*` variants take first-class dimension values in the mixed [`ArrayIrValue`](crate::ArrayIrValue)
//! family, so that extents unknown at trace time remain available as values.
//!
//! # Examples
//!
//! Static manipulation composes through the value capabilities and runs eagerly on arrays:
//!
//! ```rust
//! # use ryft_core::{Array, Concatenate, ProgramError, Reshape, Reverse, Slice, Transpose};
//! # fn main() -> Result<(), ProgramError> {
//! let matrix = Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
//! let last_two_columns = matrix.slice(&[0, 1], &[2, 3], &[1, 1])?.transpose_reversed()?;
//! let stacked = Array::concatenate([&last_two_columns, &last_two_columns], 0)?;
//! let flattened = stacked.reshape([8])?.reverse([0])?;
//! assert_eq!(flattened, Array::vector(vec![6.0_f32, 3.0, 5.0, 2.0, 6.0, 3.0, 5.0, 2.0])?);
//! # Ok(())
//! # }
//! ```
//!
//! Selections read like array indexing, and runtime start indices go through the dynamic slicing functions:
//!
//! ```rust
//! # use ryft_core::{Array, DynamicSlice, GatherOptions, Indexing, ProgramError, index};
//! # fn main() -> Result<(), ProgramError> {
//! let matrix = Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6])?;
//! let last_column = matrix.at(&index![.., -1]).get(&GatherOptions::new())?;
//! assert_eq!(last_column, Array::vector(vec![3_i32, 6])?);
//! let start = Array::scalar(-2_i32)?;
//! let window = matrix.dynamic_slice_in_axis(&start, 2, 1)?;
//! assert_eq!(window, Array::matrix(2, 2, vec![2_i32, 3, 5, 6])?);
//! # Ok(())
//! # }
//! ```
//!
//! The same selections can update references in place through a view of the selected region:
//!
//! ```rust
//! # use ryft_core::{Array, ArrayIrValue, Indexing, ProgramError, ReferenceNew, ReferenceRead, index};
//! # fn main() -> Result<(), ProgramError> {
//! let buffer = ArrayIrValue::Array(Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6])?).reference_new()?;
//! buffer.at(&index![1, 1..]).write(&ArrayIrValue::Array(Array::vector(vec![50_i32, 60])?))?;
//! assert_eq!(buffer.read()?, ArrayIrValue::Array(Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 50, 60])?));
//! # Ok(())
//! # }
//! ```

pub mod broadcasting;
pub mod concatenation;
pub mod conversions;
pub mod gathering;
pub mod indexing;
pub mod memory;
pub mod padding;
pub mod reshaping;
pub mod reversing;
pub mod scattering;
pub mod slicing;
pub mod transposition;

pub use broadcasting::{
    BROADCAST_OPERATION_NAME, Broadcast, BroadcastOperation, DynamicBroadcast, DynamicBroadcastOperation,
};
pub use concatenation::{CONCATENATE_OPERATION_NAME, Concatenate, ConcatenateOperation, DynamicConcatenate};
pub use conversions::{
    CONVERT_ELEMENT_TYPE_OPERATION_NAME, ConvertElementType, ConvertElementTypeOperation, ElementType,
};
pub use gathering::{
    DynamicGather, GATHER_OPERATION_NAME, Gather, GatherDimensionNumbers, GatherMode, GatherOperation, GatherOptions,
};
pub use indexing::{BasicIndex, IndexInteger, IndexMask, IndexSelector, IndexSlice, Indexed, Indexing};
pub use memory::{TRANSFER_TO_MEMORY_OPERATION_NAME, TransferToMemory, TransferToMemoryOperation};
pub use padding::{DynamicPad, PAD_OPERATION_NAME, Pad, PadOperation};
pub use reshaping::{DynamicReshape, DynamicReshapeOperation, RESHAPE_OPERATION_NAME, Reshape, ReshapeOperation};
pub use reversing::{REVERSE_OPERATION_NAME, Reverse, ReverseOperation};
pub use scattering::{
    DynamicScatter, SCATTER_OPERATION_NAME, Scatter, ScatterDimensionNumbers, ScatterMode, ScatterOperation,
    ScatterOptions, ScatterReductionKind,
};
pub use slicing::{
    DYNAMIC_SLICE_OPERATION_NAME, DYNAMIC_UPDATE_SLICE_OPERATION_NAME, DynamicSlice, DynamicSliceBounds,
    DynamicSliceOperation, DynamicSliceWithDimensions, DynamicUpdateSlice, DynamicUpdateSliceOperation,
    SLICE_OPERATION_NAME, Slice, SliceOperation, UPDATE_SLICE_OPERATION_NAME, UpdateSlice, UpdateSliceOperation,
};
pub use transposition::{Permutation, TRANSPOSE_OPERATION_NAME, Transpose, TransposeOperation};
