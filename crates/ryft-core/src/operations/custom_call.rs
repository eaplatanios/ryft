//! Calls to foreign kernels registered with the executing backend, which are the analogue of
//! [`jax.ffi.ffi_call`](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.ffi_call.html). This module provides the
//! following:
//!
//!   - The [`CustomCallOperation`], which carries a target name, declared output types, typed
//!     [`CustomCallAttribute`] configuration, and optional input layouts, buffer aliases, and observable
//!     [`EffectClass`]. Backends resolve the target in their kernel registry and execute the kernel. The reference
//!     [`Array`] backend has no such registry, and so it reports an unsupported-operation error.
//!   - The [`CustomCall`] capability, which executes or stages a [`CustomCallOperation`] on concrete values and
//!     transform tracers alike.
//!   - The transform contracts that an opaque kernel cannot provide by itself: [`CustomCallBatching`] selects how
//!     mapped inputs reach the kernel, and [`CustomCallRaggedContract`] describes existing packed-data and extent
//!     inputs so that ragged batches can be discharged without changing the kernel signature.
//!
//! Homogeneous array calls require static output shapes. Mixed [`ArrayIrType`] calls additionally accept trailing
//! first-class dimension inputs that ground dynamic output axes without entering the kernel ABI. Undeclared output
//! shardings inherit the manual-axis variation of the inputs. Differentiation requires wrapping the call in a
//! [`custom_function`](crate::custom_function) with derivative rules, while partial evaluation uses the ordinary
//! fold-or-residualize behavior of the executing context. Refer to [`CustomCallOperation`] for the complete calling
//! convention and to the [StableHLO specification](https://openxla.org/stablehlo/spec#custom_call) for the operation
//! that the XLA backend lowers it to.
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Array, ArrayType, CustomCall, CustomCallOperation, DataType, ProgramError};
//! let operation = CustomCallOperation::new("example.kernel", vec![ArrayType::new_static(DataType::F32, [2])])
//!     .with_attribute("scale", 2.0f32);
//! let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
//! assert!(matches!(
//!     Array::custom_call(&operation, [&input]),
//!     Err(ProgramError::UnsupportedOperation { .. }),
//! ));
//! ```

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt::Display;
use std::hash::{Hash, Hasher};

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy,
    ArrayIrType, ArrayIrValue, ArrayType, DataType, Dimension, DimensionSource, DimensionType, DimensionValue,
    DimensionVariable, Layout, RaggedAxis, Sharding, ShardingDimension, TiledLayout,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    MemberBatchableOperation, batch_projected_operation,
};
use crate::contexts::{Context, Domain, EagerContext};
use crate::differentiation::{
    DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    DifferentiationPolicy, MemberDifferentiableOperation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver, MemberInterpretableOperation};
use crate::macros::{check_count, impl_differentiable_operation, impl_reference_dischargeable_operation};
use crate::operations::Capability;
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::control_flow::scan::ScanOperation;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{BroadcastOperation, DynamicBroadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
use crate::parameters::Placeholder;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    EffectClass, EffectClasses, Effects, EmptyRegionDriver, MemberOperation, Operation, OperationFormatter,
    OperationProjection, ProgramBuilder, ProgramError, RegionInterface, Type, TypeError, TypeIdentityRenaming, Typed,
    Value, ValueProjection,
};

/// Typed configuration attribute value carried by a [`CustomCallOperation`] and forwarded to the foreign kernel.
/// The variants cover every attribute kind that a typed foreign-function interface can decode: UTF-8 strings,
/// arbitrary binary data, Booleans, signed and unsigned integers of 8, 16, 32, and 64 bits, 32-bit and 64-bit
/// floating-point values, one-dimensional [`CustomCallArrayAttribute`]s of those numeric types, and nested
/// dictionaries. Kernels decode scalars strictly by type, so a kernel that expects, for example, a 32-bit float must
/// receive [`F32`](Self::F32) rather than [`F64`](Self::F64). Binary payloads preserve every byte, including NUL and
/// invalid UTF-8, without an implicit text encoding. Backends must preserve these values or reject unsupported
/// variants.
///
/// The `From` conversions allow direct arguments to [`CustomCallOperation::with_attribute`]: `&str` and `String`
/// become [`String`](Self::String), `&[u8]` and `Vec<u8>` become [`Bytes`](Self::Bytes), every supported scalar
/// type becomes its corresponding scalar variant, and every other supported `Vec<T>` becomes an
/// [`Array`](Self::Array). Unsigned 8-bit arrays must therefore be spelled out explicitly as
/// [`CustomCallArrayAttribute::U8`]. Unsuffixed integer and floating-point literals follow Rust's default literal
/// types (i.e., `i32` and `f64`), so use suffixes such as `4i64` or `2.5f32` to select another width.
///
/// Equality and hashing compare floating-point values bitwise, so that attributes are faithful keys of the values
/// that backends receive (e.g., `-0.0` and `+0.0` are distinct, and every NaN equals itself). [`I64`](Self::I64) and
/// [`F64`](Self::F64) render as bare literals, while the other scalar variants render with a Rust type suffix
/// (e.g., `4i32` or `2.5f32`), so that renderings distinguish every scalar type.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{CustomCallArrayAttribute, CustomCallAttribute};
/// assert_eq!(CustomCallAttribute::from(2.5f32), CustomCallAttribute::F32(2.5));
/// assert_eq!(
///     CustomCallAttribute::from(vec![1i32, 2]),
///     CustomCallAttribute::Array(CustomCallArrayAttribute::I32(vec![1, 2])),
/// );
/// assert_eq!(
///     CustomCallAttribute::Dictionary(vec![
///         ("scale".to_string(), CustomCallAttribute::from(2.5f32)),
///         ("mode".to_string(), CustomCallAttribute::from("fast")),
///     ])
///     .to_string(),
///     "{scale=2.5f32, mode=fast}",
/// );
/// ```
#[derive(Clone, Debug)]
pub enum CustomCallAttribute {
    /// UTF-8 string value.
    String(String),

    /// Opaque binary data with no UTF-8 requirement.
    Bytes(Vec<u8>),

    /// Boolean value.
    Boolean(bool),

    /// 8-bit signed-integer value.
    I8(i8),

    /// 16-bit signed-integer value.
    I16(i16),

    /// 32-bit signed-integer value.
    I32(i32),

    /// 64-bit signed-integer value.
    I64(i64),

    /// 8-bit unsigned-integer value.
    U8(u8),

    /// 16-bit unsigned-integer value.
    U16(u16),

    /// 32-bit unsigned-integer value.
    U32(u32),

    /// 64-bit unsigned-integer value.
    U64(u64),

    /// 32-bit floating-point value.
    F32(f32),

    /// 64-bit floating-point value.
    F64(f64),

    /// One-dimensional array of numeric values.
    Array(CustomCallArrayAttribute),

    /// Nested dictionary of named attributes, in insertion order. Names must be unique within each dictionary.
    Dictionary(Vec<(String, CustomCallAttribute)>),
}

impl CustomCallAttribute {
    /// Returns the first attribute name that `attributes`, or any dictionary nested in them, declares more than once.
    fn duplicate_name(attributes: &[(String, Self)]) -> Option<&str> {
        attributes.iter().enumerate().find_map(|(index, (name, value))| {
            if attributes[..index].iter().any(|(existing, _)| existing == name) {
                return Some(name.as_str());
            }
            match value {
                Self::Dictionary(attributes) => Self::duplicate_name(attributes),
                _ => None,
            }
        })
    }
}

impl Display for CustomCallAttribute {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(string) => formatter.write_str(string),
            Self::Bytes(bytes) => write!(formatter, "bytes {bytes:02x?}"),
            Self::Boolean(boolean) => write!(formatter, "{boolean}"),
            Self::I8(integer) => write!(formatter, "{integer}i8"),
            Self::I16(integer) => write!(formatter, "{integer}i16"),
            Self::I32(integer) => write!(formatter, "{integer}i32"),
            Self::I64(integer) => write!(formatter, "{integer}"),
            Self::U8(integer) => write!(formatter, "{integer}u8"),
            Self::U16(integer) => write!(formatter, "{integer}u16"),
            Self::U32(integer) => write!(formatter, "{integer}u32"),
            Self::U64(integer) => write!(formatter, "{integer}u64"),
            Self::F32(float) => write!(formatter, "{float:?}f32"),
            Self::F64(float) => write!(formatter, "{float:?}"),
            Self::Array(array) => write!(formatter, "{array}"),
            Self::Dictionary(attributes) => {
                formatter.write_str("{")?;
                for (index, (name, value)) in attributes.iter().enumerate() {
                    if index > 0 {
                        formatter.write_str(", ")?;
                    }
                    write!(formatter, "{name}={value}")?;
                }
                formatter.write_str("}")
            }
        }
    }
}

impl PartialEq for CustomCallAttribute {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::String(left), Self::String(right)) => left == right,
            (Self::Bytes(left), Self::Bytes(right)) => left == right,
            (Self::Boolean(left), Self::Boolean(right)) => left == right,
            (Self::I8(left), Self::I8(right)) => left == right,
            (Self::I16(left), Self::I16(right)) => left == right,
            (Self::I32(left), Self::I32(right)) => left == right,
            (Self::I64(left), Self::I64(right)) => left == right,
            (Self::U8(left), Self::U8(right)) => left == right,
            (Self::U16(left), Self::U16(right)) => left == right,
            (Self::U32(left), Self::U32(right)) => left == right,
            (Self::U64(left), Self::U64(right)) => left == right,
            (Self::F32(left), Self::F32(right)) => left.to_bits() == right.to_bits(),
            (Self::F64(left), Self::F64(right)) => left.to_bits() == right.to_bits(),
            (Self::Array(left), Self::Array(right)) => left == right,
            (Self::Dictionary(left), Self::Dictionary(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for CustomCallAttribute {}

impl Hash for CustomCallAttribute {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::String(string) => string.hash(state),
            Self::Bytes(bytes) => bytes.hash(state),
            Self::Boolean(boolean) => boolean.hash(state),
            Self::I8(integer) => integer.hash(state),
            Self::I16(integer) => integer.hash(state),
            Self::I32(integer) => integer.hash(state),
            Self::I64(integer) => integer.hash(state),
            Self::U8(integer) => integer.hash(state),
            Self::U16(integer) => integer.hash(state),
            Self::U32(integer) => integer.hash(state),
            Self::U64(integer) => integer.hash(state),
            Self::F32(float) => float.to_bits().hash(state),
            Self::F64(float) => float.to_bits().hash(state),
            Self::Array(array) => array.hash(state),
            Self::Dictionary(attributes) => attributes.hash(state),
        }
    }
}

impl From<&str> for CustomCallAttribute {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<String> for CustomCallAttribute {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&[u8]> for CustomCallAttribute {
    fn from(value: &[u8]) -> Self {
        Self::Bytes(value.to_vec())
    }
}

impl From<Vec<u8>> for CustomCallAttribute {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

impl From<bool> for CustomCallAttribute {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl From<CustomCallArrayAttribute> for CustomCallAttribute {
    fn from(value: CustomCallArrayAttribute) -> Self {
        Self::Array(value)
    }
}

/// Implements the scalar and array `From` conversions into [`CustomCallAttribute`] for one numeric element type.
macro_rules! impl_custom_call_attribute_conversions {
    // This branch converts both scalars and vectors of the element type.
    ($type:ty, $variant:ident) => {
        impl_custom_call_attribute_conversions!(@scalar $type, $variant);

        impl From<Vec<$type>> for CustomCallAttribute {
            fn from(value: Vec<$type>) -> Self {
                Self::Array(CustomCallArrayAttribute::$variant(value))
            }
        }
    };

    // This branch converts only scalars of the element type, for `u8`, whose vectors are binary data.
    (@scalar $type:ty, $variant:ident) => {
        impl From<$type> for CustomCallAttribute {
            fn from(value: $type) -> Self {
                Self::$variant(value)
            }
        }
    };
}

impl_custom_call_attribute_conversions!(i8, I8);
impl_custom_call_attribute_conversions!(i16, I16);
impl_custom_call_attribute_conversions!(i32, I32);
impl_custom_call_attribute_conversions!(i64, I64);
impl_custom_call_attribute_conversions!(@scalar u8, U8);
impl_custom_call_attribute_conversions!(u16, U16);
impl_custom_call_attribute_conversions!(u32, U32);
impl_custom_call_attribute_conversions!(u64, U64);
impl_custom_call_attribute_conversions!(f32, F32);
impl_custom_call_attribute_conversions!(f64, F64);

/// One-dimensional numeric array carried by [`CustomCallAttribute::Array`]. The element types are exactly the numeric
/// array element types that a typed foreign-function interface can decode (e.g., as a `Span<const int32_t>` in XLA's
/// FFI). Boolean arrays are deliberately absent because those interfaces cannot decode them. Equality and hashing
/// compare floating-point elements bitwise, like [`CustomCallAttribute`] does for scalars, and arrays render with
/// their element type (e.g., `array<i32: 1, 2>`).
#[derive(Clone, Debug)]
pub enum CustomCallArrayAttribute {
    /// Array of 8-bit signed integers.
    I8(Vec<i8>),

    /// Array of 16-bit signed integers.
    I16(Vec<i16>),

    /// Array of 32-bit signed integers.
    I32(Vec<i32>),

    /// Array of 64-bit signed integers.
    I64(Vec<i64>),

    /// Array of 8-bit unsigned integers.
    U8(Vec<u8>),

    /// Array of 16-bit unsigned integers.
    U16(Vec<u16>),

    /// Array of 32-bit unsigned integers.
    U32(Vec<u32>),

    /// Array of 64-bit unsigned integers.
    U64(Vec<u64>),

    /// Array of 32-bit floating-point numbers.
    F32(Vec<f32>),

    /// Array of 64-bit floating-point numbers.
    F64(Vec<f64>),
}

impl CustomCallArrayAttribute {
    /// Returns the number of elements in this array.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::I8(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    /// Returns `true` if this array has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Display for CustomCallArrayAttribute {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn write_values<T: std::fmt::Debug>(
            formatter: &mut std::fmt::Formatter<'_>,
            element_type: &str,
            values: &[T],
        ) -> std::fmt::Result {
            write!(formatter, "array<{element_type}")?;
            for (index, value) in values.iter().enumerate() {
                write!(formatter, "{}{value:?}", if index == 0 { ": " } else { ", " })?;
            }
            formatter.write_str(">")
        }

        match self {
            Self::I8(values) => write_values(formatter, "i8", values),
            Self::I16(values) => write_values(formatter, "i16", values),
            Self::I32(values) => write_values(formatter, "i32", values),
            Self::I64(values) => write_values(formatter, "i64", values),
            Self::U8(values) => write_values(formatter, "u8", values),
            Self::U16(values) => write_values(formatter, "u16", values),
            Self::U32(values) => write_values(formatter, "u32", values),
            Self::U64(values) => write_values(formatter, "u64", values),
            Self::F32(values) => write_values(formatter, "f32", values),
            Self::F64(values) => write_values(formatter, "f64", values),
        }
    }
}

impl PartialEq for CustomCallArrayAttribute {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::I8(left), Self::I8(right)) => left == right,
            (Self::I16(left), Self::I16(right)) => left == right,
            (Self::I32(left), Self::I32(right)) => left == right,
            (Self::I64(left), Self::I64(right)) => left == right,
            (Self::U8(left), Self::U8(right)) => left == right,
            (Self::U16(left), Self::U16(right)) => left == right,
            (Self::U32(left), Self::U32(right)) => left == right,
            (Self::U64(left), Self::U64(right)) => left == right,
            (Self::F32(left), Self::F32(right)) => {
                left.iter().map(|value| value.to_bits()).eq(right.iter().map(|value| value.to_bits()))
            }
            (Self::F64(left), Self::F64(right)) => {
                left.iter().map(|value| value.to_bits()).eq(right.iter().map(|value| value.to_bits()))
            }
            _ => false,
        }
    }
}

impl Eq for CustomCallArrayAttribute {}

impl Hash for CustomCallArrayAttribute {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::I8(values) => values.hash(state),
            Self::I16(values) => values.hash(state),
            Self::I32(values) => values.hash(state),
            Self::I64(values) => values.hash(state),
            Self::U8(values) => values.hash(state),
            Self::U16(values) => values.hash(state),
            Self::U32(values) => values.hash(state),
            Self::U64(values) => values.hash(state),
            Self::F32(values) => {
                values.len().hash(state);
                values.iter().for_each(|value| value.to_bits().hash(state));
            }
            Self::F64(values) => {
                values.len().hash(state);
                values.iter().for_each(|value| value.to_bits().hash(state));
            }
        }
    }
}

/// Declares that one flat array output of a [`CustomCallOperation`] aliases one flat array input.
///
/// Aliasing requires the input and output to describe the same logical array. It allows a backend to reuse the input
/// buffer for the output without changing Ryft's functional SSA semantics. The indices never include mixed trailing
/// dimension inputs or backend-internal effect tokens.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomCallInputOutputAlias {
    /// Index of the aliased array input.
    input_index: usize,

    /// Index of the array output that aliases the input.
    output_index: usize,
}

impl CustomCallInputOutputAlias {
    /// Creates an alias from the array input at `input_index` to the array output at `output_index`.
    #[inline]
    pub fn new(input_index: usize, output_index: usize) -> Self {
        Self { input_index, output_index }
    }

    /// Returns the index of the aliased array input.
    #[inline]
    pub fn input_index(&self) -> usize {
        self.input_index
    }

    /// Returns the index of the array output that aliases the input.
    #[inline]
    pub fn output_index(&self) -> usize {
        self.output_index
    }
}

impl Display for CustomCallInputOutputAlias {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}->{}", self.input_index, self.output_index)
    }
}

/// Behavior a [`CustomCallOperation`] requests when the batching transform maps one of its inputs. Ryft cannot derive a
/// batching rule for an opaque kernel, so the author of the call declares which of the few universally meaningful
/// strategies applies, using [`with_batching`](CustomCallOperation::with_batching). This mirrors JAX's `vmap_method`
/// selection on [`jax.ffi.ffi_call`](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.ffi_call.html). A call whose
/// inputs are all replicated never consults this behavior: it is bound unchanged.
///
/// The single-call strategies require the kernel to understand batch-prefixed buffers. Every output gains the full
/// mapped extent, so an input/output alias remains valid only when its input also carries that complete output shape.
/// Aliases of replicated inputs are consequently rejected by [`ExpandDimensions`](Self::ExpandDimensions) when the
/// batch extent exceeds one, and by [`Vectorized`](Self::Vectorized) whenever they would change the logical shape.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CustomCallBatching {
    /// Report a [`BatchingError::UnsupportedOperation`] naming the mapped input. This is the default because a foreign
    /// kernel's contract is opaque: silently choosing a strategy could execute the kernel on buffers it never agreed to
    /// accept.
    #[default]
    Rejected,

    /// Apply the kernel once per batch item through a `scan` whose body performs exactly one unbatched call. Mapped
    /// inputs are realigned to batch axis `0` and sliced one row per iteration, replicated inputs become invariant loop
    /// carries, and the per-iteration results are stacked back on batch axis `0`. The kernel therefore observes exactly
    /// the buffers it would have seen without the transform, at the cost of `b` sequential calls; a side-effecting
    /// kernel consequently runs `b` ordered times. The optional `unroll` factor is forwarded to
    /// [`ScanOperation::with_unroll`], and is a lowering-only knob that trades code size for loop overhead.
    Sequential {
        /// Lowering-only number of body copies emitted per loop trip, or [`None`] to keep one call per trip. The
        /// factor must be at least `1`.
        unroll: Option<usize>,
    },

    /// Apply the kernel once per batch item exactly like [`Sequential`](Self::Sequential), but fully unroll the
    /// staged `scan` so that lowerings emit one straight-line kernel call per batch item and no loop at all. This is
    /// the `sequential_unrolled` calling convention of `jax.ffi.ffi_call`. Full unrolling requires a statically known
    /// batch extent, so batching over a dynamic extent reports a [`BatchingError::UnsupportedOperation`].
    SequentialUnrolled,

    /// Align every input to batch axis `0` and call the kernel exactly once on batch-prefixed buffers, declaring
    /// batch-prefixed output types. The kernel must itself understand the leading batch axis. Replicated inputs are
    /// materialized across the batch first, so every input and result carries the same leading extent, aliases stay
    /// type-preserving, and a side-effecting kernel runs exactly once.
    BroadcastAll,

    /// Align mapped inputs to batch axis `0`, insert a size-one leading axis into replicated inputs, and call the
    /// kernel exactly once. Every output gains the full mapped extent. The kernel must broadcast its size-one inputs
    /// internally. This is the `expand_dims` calling convention of `jax.ffi.ffi_call`.
    ExpandDimensions,

    /// Align mapped inputs to batch axis `0`, keep replicated inputs unchanged, and call the kernel exactly once.
    /// Every output gains the full mapped extent. The kernel must distinguish batch-prefixed inputs from invariant
    /// inputs and implement the required vectorization itself. This is the `legacy_vectorized` calling convention of
    /// `jax.ffi.ffi_call`.
    Vectorized,
}

impl Display for CustomCallBatching {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected => formatter.write_str("rejected"),
            Self::Sequential { unroll: None } => formatter.write_str("sequential"),
            Self::Sequential { unroll: Some(unroll) } => write!(formatter, "sequential(unroll={unroll})"),
            Self::SequentialUnrolled => formatter.write_str("sequential_unrolled"),
            Self::BroadcastAll => formatter.write_str("broadcast_all"),
            Self::ExpandDimensions => formatter.write_str("expand_dimensions"),
            Self::Vectorized => formatter.write_str("vectorized"),
        }
    }
}

/// Names one packed custom-call input axis whose live extent is carried by another, already-declared input. The bound
/// [`DimensionVariable`] gives the ragged dimension stable identity across transformation replays, while
/// `extent_input_index` identifies the ordinary integer scalar input that the unbatched foreign kernel receives.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomCallRaggedInputBinding {
    /// Name used by output bindings to refer to this input binding.
    name: String,

    /// Index of the packed array input.
    input_index: usize,

    /// Axis of the packed array input whose live prefix is ragged.
    axis: usize,

    /// Index of the existing scalar integer input carrying the live extent.
    extent_input_index: usize,

    /// Stable identity and runtime bounds of the ragged dimension.
    dimension: DimensionVariable,
}

impl CustomCallRaggedInputBinding {
    /// Creates a named input binding between one packed input axis and one existing scalar extent input.
    pub fn new<N: Into<String>>(
        name: N,
        input_index: usize,
        axis: usize,
        extent_input_index: usize,
        dimension: DimensionVariable,
    ) -> Self {
        Self { name: name.into(), input_index, axis, extent_input_index, dimension }
    }

    /// Returns the binding name used by output bindings.
    #[inline]
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Returns the index of the packed array input.
    #[inline]
    pub fn input_index(&self) -> usize {
        self.input_index
    }

    /// Returns the bound axis of the packed array input.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the index of the existing input carrying the live extent.
    #[inline]
    pub fn extent_input_index(&self) -> usize {
        self.extent_input_index
    }

    /// Returns the stable identity and runtime bounds of the ragged dimension.
    #[inline]
    pub fn dimension(&self) -> &DimensionVariable {
        &self.dimension
    }
}

impl Display for CustomCallRaggedInputBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}:input({})@{}<=input({}):{}",
            self.name, self.input_index, self.axis, self.extent_input_index, self.dimension,
        )
    }
}

/// Declares how one positional custom-call output relates to the operation's ragged input bindings.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CustomCallRaggedOutputBinding {
    /// Preserve the named input binding on the output axis `axis`, reusing the exact same extent value and dimension
    /// identity. The axis may be relocated when the output does not alias its input.
    Preserved {
        /// Name of the input binding being preserved.
        input_binding: String,

        /// Packed output axis carrying the preserved ragged dimension.
        axis: usize,
    },

    /// Produce a dense output. Every input binding not preserved by any output is considered deliberately consumed by
    /// the call's declared padding-independent semantics.
    Consumed,

    /// Produce a ragged output whose live extents come from another, ordinary integer scalar output of the same call.
    Fresh {
        /// Packed output axis whose live prefix is described by the fresh extents.
        axis: usize,

        /// Index of the existing integer scalar output carrying the live extent.
        extent_output_index: usize,

        /// Stable identity and runtime bounds of the fresh ragged dimension.
        dimension: DimensionVariable,
    },
}

impl Display for CustomCallRaggedOutputBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preserved { input_binding, axis } => write!(formatter, "preserve({input_binding})@{axis}"),
            Self::Consumed => formatter.write_str("consume"),
            Self::Fresh { axis, extent_output_index, dimension } => {
                write!(formatter, "fresh@{axis}<=output({extent_output_index}):{dimension}")
            }
        }
    }
}

/// Declared calling convention that lets [`CustomCallOperation`] discharge one level of ragged batching without
/// changing the foreign kernel's signature. Input bindings point at extent inputs that already exist in the call, and
/// output bindings either preserve one of those bindings, consume raggedness, or point at an existing extent output.
/// Declaring this contract promises that no live output element depends on padded input elements.
///
/// The contract intentionally supports one ragged axis per input and one ragged batching level. A second ragged
/// batching level is rejected because representing it requires associating each ragged axis with another independent
/// extent value. Ordinary dense single-call batching remains composable: the contract records every accumulated
/// leading dense axis so that its existing extent inputs and outputs retain their complete axis mapping. The
/// declaration supplies no differentiation semantics: ragged custom calls continue to follow the ordinary custom-call
/// differentiation contract. Padding remains unspecified downstream; the result [`RaggedAxis`] metadata is what lets
/// extent-aware operations avoid observing it.
///
/// Runtime extent values must lie within the declared [`DimensionVariable`] bounds and must not exceed their packed
/// physical axis. Eager foreign-kernel implementations are responsible for checking that precondition when decoding
/// their ordinary extent inputs and outputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomCallRaggedContract {
    /// Named packed-input bindings, in declaration order.
    input_bindings: Vec<CustomCallRaggedInputBinding>,

    /// Positional output bindings, with exactly one entry per declared output.
    output_bindings: Vec<CustomCallRaggedOutputBinding>,

    /// Number of leading dense batch axes accumulated by repeated single-call batching.
    batch_prefix_count: usize,

    /// Whether an earlier batching pass discharged active ragged input bindings.
    ragged_discharged: bool,
}

impl CustomCallRaggedContract {
    /// Creates a ragged calling convention over existing custom-call inputs and outputs.
    ///
    /// # Parameters
    ///
    ///   - `input_bindings`: Named packed-input axes and their existing scalar extent inputs.
    ///   - `output_bindings`: One positional relationship for every declared custom-call output.
    pub fn new(
        input_bindings: Vec<CustomCallRaggedInputBinding>,
        output_bindings: Vec<CustomCallRaggedOutputBinding>,
    ) -> Self {
        Self { input_bindings, output_bindings, batch_prefix_count: 0, ragged_discharged: false }
    }

    /// Returns the named packed-input bindings in declaration order.
    #[inline]
    pub fn input_bindings(&self) -> &[CustomCallRaggedInputBinding] {
        self.input_bindings.as_slice()
    }

    /// Returns the positional output bindings.
    #[inline]
    pub fn output_bindings(&self) -> &[CustomCallRaggedOutputBinding] {
        self.output_bindings.as_slice()
    }

    /// Returns a contract describing the rewritten call after bound inputs and outputs gain another leading dense
    /// batch dimension.
    fn batch_prefixed(&self, discharges_ragged: bool) -> Self {
        let mut contract = self.clone();
        for binding in &mut contract.input_bindings {
            binding.axis += 1;
        }
        for binding in &mut contract.output_bindings {
            match binding {
                CustomCallRaggedOutputBinding::Preserved { axis, .. }
                | CustomCallRaggedOutputBinding::Fresh { axis, .. } => *axis += 1,
                CustomCallRaggedOutputBinding::Consumed => {}
            }
        }
        contract.batch_prefix_count += 1;
        contract.ragged_discharged |= discharges_ragged;
        contract
    }

    /// Returns a contract recording that active ragged bindings were discharged without changing input or output axes.
    /// This transition is used by `Sequential`, whose scan body sees one unbatched slice at a time.
    fn ragged_discharged(&self) -> Self {
        let mut contract = self.clone();
        contract.ragged_discharged = true;
        contract
    }

    /// Returns the unique active dimensions that are absent from every preserved ragged output.
    fn consumed_dimensions<V>(&self, active: &[(String, RaggedAxis<V>)]) -> Vec<DimensionVariable> {
        let mut consumed = Vec::new();
        for (_, axis) in active {
            let dimension = axis.dimension();
            let preserved = active.iter().any(|(name, candidate)| {
                candidate.dimension() == dimension
                    && self.output_bindings.iter().any(|binding| {
                        matches!(
                            binding,
                            CustomCallRaggedOutputBinding::Preserved { input_binding, .. }
                                if input_binding == name
                        )
                    })
            });
            if !preserved && !consumed.contains(dimension) {
                consumed.push(dimension.clone());
            }
        }
        consumed
    }

    /// Returns the complete extent-axis mapping for a new ragged level whose packed data and extent input carry the
    /// current mapped axis at `data_batch_axis` and `extent_batch_axis`, respectively.
    fn active_extent_axes(&self, data_batch_axis: usize, extent_batch_axis: usize) -> Vec<usize> {
        (0..=self.batch_prefix_count)
            .map(|extent_axis| {
                if extent_axis == extent_batch_axis {
                    data_batch_axis
                } else {
                    let prefix_axis = extent_axis - usize::from(extent_axis > extent_batch_axis);
                    prefix_axis + usize::from(prefix_axis >= data_batch_axis)
                }
            })
            .collect()
    }

    /// Returns this contract after renaming every dimension identity.
    fn renamed(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        let mut contract = self.clone();
        for binding in &mut contract.input_bindings {
            binding.dimension =
                DimensionType::from(binding.dimension.clone()).rename_identities(renaming)?.variable().clone();
        }
        for binding in &mut contract.output_bindings {
            if let CustomCallRaggedOutputBinding::Fresh { dimension, .. } = binding {
                *dimension = DimensionType::from(dimension.clone()).rename_identities(renaming)?.variable().clone();
            }
        }
        Ok(contract)
    }
}

impl Display for CustomCallRaggedContract {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("{inputs=[")?;
        for (index, binding) in self.input_bindings.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{binding}")?;
        }
        formatter.write_str("], outputs=[")?;
        for (index, binding) in self.output_bindings.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{binding}")?;
        }
        formatter.write_str("]")?;
        if self.batch_prefix_count != 0 {
            write!(formatter, ", batch_prefix_count={}", self.batch_prefix_count)?;
        }
        if self.ragged_discharged {
            formatter.write_str(", ragged_discharged=true")?;
        }
        formatter.write_str("}")
    }
}

/// Universe-neutral view of one custom-call array input during ragged contract validation.
struct CustomCallRaggedInput<'o, V> {
    /// Packed value used to verify the declared extent input.
    value: &'o V,

    /// Normalized position of the mapped batch axis.
    batch_axis: Option<usize>,

    /// Bounded ragged axes carried by the input.
    ragged_axes: &'o [RaggedAxis<V>],

    /// Physical per-item array type expected by the foreign kernel.
    physical_type: ArrayType,
}

/// Canonical operation name for [`CustomCallOperation`].
pub const CUSTOM_CALL_OPERATION_NAME: &str = "custom_call";

/// [`Operation`] that calls a foreign kernel registered with the executing backend under a target name. It is the
/// analogue of [`jax.ffi.ffi_call`](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.ffi_call.html). The operation
/// is opaque to Ryft: its output types are declared up front instead of inferred, and typed [`CustomCallAttribute`]s
/// are forwarded verbatim to the kernel as its configuration.
///
/// This is an [`ArrayType`] operation that participates in two input contracts. In homogeneous array programs it
/// accepts only the foreign kernel's array inputs, so every declared output must have a static shape. As a mixed member
/// of the array IR (through [`MemberOperation<ArrayIrType>`]) it additionally accepts one trailing first-class
/// dimension input per dynamic axis occurrence in the declared outputs, ordered first by output and then by axis, and
/// type inference verifies that each input defines the exact variable referenced by its corresponding output axis.
/// These logical result extents do not enter the foreign kernel ABI: only the leading array inputs are passed to the
/// kernel. Eager execution and backend lowering use the trailing inputs to verify or attach the declared logical sizes
/// to the returned buffers. Both contracts share one payload with the same target, output declarations, attributes,
/// layouts, aliases, effects, rendering, and backend-kernel semantics.
///
/// Type inference also validates the configuration: attribute names must be unique (including within nested
/// dictionaries), declared input layouts must match the rank of their inputs, each alias must connect an input and an
/// output of identical types (and identical declared layouts), and any [`CustomCallRaggedContract`] must be consistent
/// with the complete array signature. Output types declared without a sharding inherit the manual-axis variation of
/// the inputs, placed replicated on their mesh, because opaque code cannot be assumed to produce the same result on
/// every device.
///
/// The XLA backend lowers this operation to a
/// [`stablehlo.custom_call`](https://openxla.org/stablehlo/spec#custom_call) using the typed FFI calling convention
/// (`api_version = 4`), with the attributes carried as the `backend_config` dictionary. Handlers are registered with
/// the executing PJRT client under the same target name (e.g., via `ryft-pjrt`'s `Client::register_ffi_handler`).
/// The reference array backend cannot execute foreign kernels, so eager interpretation on it reports an error.
///
/// # Transformations
///
/// Because the kernel is opaque, Ryft cannot derive its transform rules:
///
///   - **Differentiation:** Forward-mode differentiation binds the call unchanged and returns structural zero
///     tangents when every input tangent is a structural zero (including when the call has no inputs). Any other
///     derivative reports an error directing users to call the kernel through a
///     [`custom_function`](crate::custom_function) with derivative rules (refer to
///     [`CustomFunction::from_custom_call`](crate::CustomFunction::from_custom_call)). The call is not linear, so it
///     never transposes.
///   - **Batching:** A call whose inputs are all replicated is bound unchanged, because a region-free foreign kernel
///     cannot observe the transform's named axis. A mapped input is governed by the [`CustomCallBatching`] behavior
///     selected with [`with_batching`](Self::with_batching): the default [`Rejected`](CustomCallBatching::Rejected)
///     reports an error naming that input, [`Sequential`](CustomCallBatching::Sequential) applies the kernel once per
///     batch item through a `scan` (which [`SequentialUnrolled`](CustomCallBatching::SequentialUnrolled) fully
///     unrolls), and [`BroadcastAll`](CustomCallBatching::BroadcastAll),
///     [`ExpandDimensions`](CustomCallBatching::ExpandDimensions), and [`Vectorized`](CustomCallBatching::Vectorized)
///     hand the kernel batch-prefixed buffers in a single call. A [`custom_function`](crate::custom_function) without
///     its own batching rule batches its primal region structurally, so a mapped input reaches this same operation
///     and meets this same contract. Single-call strategies shift declared input and output layouts so that the new
///     batch axis is the most major one.
///   - **Ragged batching:** A call that uses explicit packed buffers and ordinary scalar extent inputs may declare a
///     [`CustomCallRaggedContract`] with [`with_ragged_contract`](Self::with_ragged_contract). The declaration never
///     adds, removes, hides, or reorders inputs or outputs. It only lets batching verify that the exact extent value
///     attached to an input [`RaggedAxis`] is already present at the declared input index, use the selected
///     [`CustomCallBatching`] strategy unchanged, and attach preserved or fresh ragged metadata to the declared
///     outputs. Calls without this declaration reject ragged inputs.
///   - **Partial evaluation:** Known calls execute or stage through the parent context when it supports them, and pure
///     calls remain residual when an eager parent cannot execute them. Residual calls retain their declared effects
///     through dead-code elimination.
///
/// # Effects
///
/// A call is pure unless it declares an [`EffectClass`]. [`with_side_effect`](Self::with_side_effect) declares
/// [`EffectClass::OrderedIo`], which keeps the call alive through dead-code elimination and preserves its execution
/// order relative to other ordered I/O effects across every participating device.
/// [`with_effect_class`](Self::with_effect_class) selects a different contract: [`EffectClass::DeviceOrderedIo`]
/// preserves program order only among the ordered I/O executing on the same device, which is what permits the call to
/// execute once per device inside `shard_map` bodies, and [`EffectClass::UnorderedIo`] keeps the call observable
/// without any ordering dependency (the contract of JAX's `has_side_effect=True`). Every effectful call is lowered
/// with `has_side_effect = true`.
///
/// # Backend Contract
///
/// This operation is backend-independent by design, and this payload is the entire portable contract: a target
/// name resolved in the executing backend's kernel registry, declared output types, typed configuration
/// attributes, optional input layouts and buffer aliases, and an optional effect class. A backend supports the
/// operation by providing (1) a process- or client-level registry that resolves target names to executable kernels at
/// execution time, (2) a calling convention that hands the kernel its input buffers (in their declared layouts),
/// output buffers matching the declared output types, and the decoded attributes, and (3) an execution engine that
/// honors the declared effect class for side-effecting calls. Backends that cannot execute foreign kernels (like the
/// reference array backend) reject interpretation with a clear error instead of guessing.
///
/// Array layouts come from the canonical [`ArrayType`] descriptors of the results and from the declared input layouts
/// (or the canonical descriptors of the inputs, for inputs without a declaration). Portable flat-array buffer aliases
/// are declared with [`CustomCallInputOutputAlias`]. Backend-specific vocabulary must never grow on this payload:
/// encodings such as XLA's FFI API version, `backend_config` representation, tuple alias paths, result tiling
/// attributes, or called-computation references belong in the owning backend's lowering (or in a backend-owned
/// operation). If a configuration knob only makes sense for one backend, it does not belong on this operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomCallOperation {
    /// Name under which the foreign kernel is registered with the executing backend.
    target_name: String,

    /// Declared output types of the call; inference adds inherited manual variation to undeclared shardings.
    output_types: Vec<ArrayType>,

    /// Typed configuration attributes forwarded to the kernel, in insertion order.
    attributes: Vec<(String, CustomCallAttribute)>,

    /// Declared buffer layouts of the array inputs, with one entry per array input, or empty when undeclared.
    input_layouts: Vec<Option<Layout>>,

    /// Flat array input/output buffer aliases, in declaration order.
    input_output_aliases: Vec<CustomCallInputOutputAlias>,

    /// Observable effect of the call, or `None` for a pure call.
    effect_class: Option<EffectClass>,

    /// Behavior requested when the batching transform maps one of this call's inputs.
    batching: CustomCallBatching,

    /// Optional declared calling convention for discharging one level of ragged batching.
    ragged_contract: Option<CustomCallRaggedContract>,
}

impl CustomCallOperation {
    /// Creates a new [`CustomCallOperation`] with the provided target name and declared output types.
    ///
    /// # Parameters
    ///
    ///   - `target_name`: Name under which the foreign kernel is registered with the executing backend.
    ///   - `output_types`: Declared output types; undeclared shardings inherit the inputs' manual variation.
    #[inline]
    pub fn new<N: Into<String>>(target_name: N, output_types: Vec<ArrayType>) -> Self {
        Self {
            target_name: target_name.into(),
            output_types,
            attributes: Vec::new(),
            input_layouts: Vec::new(),
            input_output_aliases: Vec::new(),
            effect_class: None,
            batching: CustomCallBatching::default(),
            ragged_contract: None,
        }
    }

    /// Returns this [`CustomCallOperation`] with the provided typed configuration attribute appended. Attribute names
    /// must be unique, which type inference validates (including within nested
    /// [`Dictionary`](CustomCallAttribute::Dictionary) attributes).
    #[inline]
    pub fn with_attribute<N: Into<String>, V: Into<CustomCallAttribute>>(mut self, name: N, value: V) -> Self {
        self.attributes.push((name.into(), value.into()));
        self
    }

    /// Returns this [`CustomCallOperation`] declaring the buffer layouts in which the kernel receives its array
    /// inputs, replacing any earlier declaration. The declaration has one entry per array input (excluding the
    /// trailing dimension inputs of the mixed [`ArrayIrType`] form). [`None`] entries leave the corresponding input in
    /// the layout of its own [`ArrayType`], and an empty declaration leaves every input that way. Backends convert each
    /// declared input to its declared layout before invoking the kernel, so the kernel's buffer contract does not
    /// depend on how the input value happens to be stored. This holds after batching too, which shifts declared
    /// layouts together with the inputs that gain a leading batch axis. Count, rank, and alias compatibility are
    /// validated during type inference, when the input types are available.
    #[inline]
    pub fn with_input_layouts<L: IntoIterator<Item = Option<Layout>>>(mut self, input_layouts: L) -> Self {
        self.input_layouts = input_layouts.into_iter().collect();
        self
    }

    /// Returns this operation with an alias from array input `input_index` to array output `output_index`.
    ///
    /// Index bounds and type compatibility are validated during type inference, when the input types are available.
    /// Each input and output can participate in at most one alias.
    pub fn with_input_output_alias(mut self, input_index: usize, output_index: usize) -> Result<Self, TypeError> {
        if let Some(alias) = self
            .input_output_aliases
            .iter()
            .find(|alias| alias.input_index == input_index || alias.output_index == output_index)
        {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` cannot add alias {input_index}->{output_index} because alias \
                 `{alias}` already uses the same input or output",
            )));
        }
        self.input_output_aliases.push(CustomCallInputOutputAlias::new(input_index, output_index));
        Ok(self)
    }

    /// Returns this call declaring [`EffectClass::OrderedIo`], replacing any earlier effect selection. This is the
    /// strongest I/O contract: one execution order across every participating device, which restricts the program
    /// to single-device placement. Refer to [`with_effect_class`](Self::with_effect_class) for the alternatives.
    #[inline]
    pub fn with_side_effect(self) -> Self {
        self.with_effect_class(EffectClass::OrderedIo)
    }

    /// Returns this call declaring the provided observable effect class, replacing any earlier selection.
    ///
    /// Every class keeps the call alive through dead-code elimination and marks the lowered call `has_side_effect`.
    /// The I/O classes differ in ordering scope: [`EffectClass::OrderedIo`] promises one order across devices,
    /// [`EffectClass::DeviceOrderedIo`] promises program order only among the ordered I/O executing on the same
    /// device and therefore permits the handler to run once per device (the handler must tolerate that
    /// replication), and [`EffectClass::UnorderedIo`] promises no ordering. The effect class does not supply
    /// batching or differentiation rules, and does not determine whether execution blocks the calling thread.
    #[inline]
    pub fn with_effect_class(mut self, effect_class: EffectClass) -> Self {
        self.effect_class = Some(effect_class);
        self
    }

    /// Returns this [`CustomCallOperation`] requesting the provided [`CustomCallBatching`] behavior when the batching
    /// transform maps one of its inputs. Refer to the documentation of [`CustomCallBatching`] for the available
    /// behaviors and for why the default rejects mapped inputs.
    #[inline]
    pub fn with_batching(mut self, batching: CustomCallBatching) -> Self {
        self.batching = batching;
        self
    }

    /// Returns this operation carrying the provided declared ragged calling convention. Structural validation occurs
    /// during type inference, when the input types are available.
    #[inline]
    pub fn with_ragged_contract(mut self, contract: CustomCallRaggedContract) -> Self {
        self.ragged_contract = Some(contract);
        self
    }

    /// Returns the name under which the foreign kernel is registered with the executing backend.
    #[inline]
    pub fn target_name(&self) -> &str {
        self.target_name.as_str()
    }

    /// Returns the declared output types of the call.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        self.output_types.as_slice()
    }

    /// Returns the typed configuration attributes forwarded to the kernel, in insertion order.
    #[inline]
    pub fn attributes(&self) -> &[(String, CustomCallAttribute)] {
        self.attributes.as_slice()
    }

    /// Returns the declared buffer layouts of the array inputs, which are either empty or contain one entry per array
    /// input. Refer to [`with_input_layouts`](Self::with_input_layouts) for more information.
    #[inline]
    pub fn input_layouts(&self) -> &[Option<Layout>] {
        self.input_layouts.as_slice()
    }

    /// Returns the flat array input/output buffer aliases in declaration order.
    #[inline]
    pub fn input_output_aliases(&self) -> &[CustomCallInputOutputAlias] {
        self.input_output_aliases.as_slice()
    }

    /// Returns whether the call has observable side effects beyond its returned outputs.
    #[inline]
    pub fn has_side_effect(&self) -> bool {
        self.effect_class.is_some()
    }

    /// Returns the observable [`EffectClass`] declared by this call, or `None` for a pure call.
    #[inline]
    pub fn effect_class(&self) -> Option<EffectClass> {
        self.effect_class
    }

    /// Returns the [`CustomCallBatching`] behavior requested when the batching transform maps one of this call's
    /// inputs.
    #[inline]
    pub fn batching(&self) -> CustomCallBatching {
        self.batching
    }

    /// Returns the declared ragged calling convention, when present.
    #[inline]
    pub fn ragged_contract(&self) -> Option<&CustomCallRaggedContract> {
        self.ragged_contract.as_ref()
    }

    /// Returns this payload with every declared output identity renamed according to `renaming`.
    fn renamed(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self {
            target_name: self.target_name.clone(),
            output_types: self
                .output_types
                .iter()
                .map(|r#type| r#type.rename_identities(renaming))
                .collect::<Result<Vec<_>, _>>()?,
            attributes: self.attributes.clone(),
            input_layouts: self.input_layouts.clone(),
            input_output_aliases: self.input_output_aliases.clone(),
            effect_class: self.effect_class,
            batching: self.batching,
            ragged_contract: self.ragged_contract.as_ref().map(|contract| contract.renamed(renaming)).transpose()?,
        })
    }

    /// Returns the number of trailing first-class output-extent inputs this call consumes in the mixed universe, which
    /// is one per dynamic axis occurrence across its declared outputs.
    fn dynamic_output_dimension_count(&self) -> usize {
        self.output_types
            .iter()
            .flat_map(|output_type| output_type.shape().dimensions())
            .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
            .count()
    }

    /// Returns this call's declared output types with a leading batch dimension inserted, which is the declaration a
    /// single-call batching strategy hands to its kernel.
    ///
    /// An output that aliases an input takes the aligned input's packed type verbatim when it has the complete
    /// batch-prefixed shape. Otherwise the output keeps its full batch extent and normal alias validation rejects
    /// the incompatible input shape. Every other output inserts the batch dimension itself as its most major axis,
    /// shifting any explicit layout as described in [`batch_prefixed_layout`](Self::batch_prefixed_layout).
    ///
    /// # Parameters
    ///
    ///   - `aligned_input_types`: Packed types of this call's array inputs after alignment to the batch axis.
    ///   - `batch_dimension`: Mapped-axis [`Dimension`] inserted as each output's new leading axis.
    ///   - `axis_sharding`: Placement assigned to the inserted axis of outputs that carry sharding metadata.
    fn batch_prefixed_output_types(
        &self,
        aligned_input_types: &[ArrayType],
        batch_dimension: Dimension,
        axis_sharding: &ShardingDimension,
    ) -> Result<Vec<ArrayType>, BatchingError> {
        self.output_types
            .iter()
            .enumerate()
            .map(|(output_index, output_type)| {
                let batched_type = output_type.batched(0, batch_dimension.clone(), axis_sharding.clone())?;
                if let Some(aligned_type) = self
                    .input_output_aliases
                    .iter()
                    .find(|alias| alias.output_index == output_index)
                    .and_then(|alias| aligned_input_types.get(alias.input_index))
                    .filter(|aligned_type| aligned_type.shape() == batched_type.shape())
                {
                    return Ok(aligned_type.clone());
                }
                Ok(match output_type.layout() {
                    None => batched_type,
                    Some(layout) => {
                        batched_type.with_layout(self.batch_prefixed_layout("output", output_index, layout)?)
                    }
                })
            })
            .collect()
    }

    /// Returns the layout required by an aliased input at its unchanged or batch-prefixed logical shape. An
    /// incompatible shape is left to alias validation, rather than changing geometry to satisfy an alias.
    fn alias_input_layout(
        &self,
        input_index: usize,
        input_type: &ArrayType,
        batch_dimension: Option<&Dimension>,
    ) -> Result<Option<Layout>, BatchingError> {
        let Some(alias) = self.input_output_aliases.iter().find(|alias| alias.input_index == input_index) else {
            return Ok(None);
        };
        let Some(output_type) = self.output_types.get(alias.output_index) else {
            return Ok(None);
        };
        let expected_type = match batch_dimension {
            Some(dimension) => output_type.with_inserted_dimension(0, dimension.clone())?,
            None => output_type.clone(),
        };
        if input_type.shape() != expected_type.shape() {
            return Ok(None);
        }
        Ok(match (output_type.layout(), batch_dimension) {
            (None, _) => None,
            (Some(layout), None) => Some(layout.clone()),
            (Some(layout), Some(_)) => Some(self.batch_prefixed_layout("output", alias.output_index, layout)?),
        })
    }

    /// Returns this call's declared input layouts for a single-call batching strategy, in which the inputs selected by
    /// `batch_prefixed` gain a leading batch axis and their declared layouts shift accordingly. Refer to
    /// [`batch_prefixed_layout`](Self::batch_prefixed_layout) for how layouts shift.
    fn batch_prefixed_input_layouts<F: Fn(usize) -> bool>(
        &self,
        batch_prefixed: F,
    ) -> Result<Vec<Option<Layout>>, BatchingError> {
        self.input_layouts
            .iter()
            .enumerate()
            .map(|(input_index, layout)| match layout {
                Some(layout) if batch_prefixed(input_index) => {
                    self.batch_prefixed_layout("input", input_index, layout).map(Some)
                }
                layout => Ok(layout.clone()),
            })
            .collect()
    }

    /// Returns `layout` after inserting a new most-major batch axis into the array it describes. An explicit
    /// [`TiledLayout`] shifts each logical dimension index by one and gains the new axis as its most major physical
    /// dimension, which keeps its tiles valid because tiling applies to the most minor dimensions. Layouts are part of
    /// the foreign kernel's buffer contract and must not be silently dropped, so a
    /// [`StridedLayout`](crate::arrays::StridedLayout) is rejected instead, because a correct batch stride depends on
    /// element sizes that the layout does not carry.
    ///
    /// # Parameters
    ///
    ///   - `role`: Either `"input"` or `"output"`, naming the role of the laid-out array in diagnostics.
    ///   - `index`: Index of the laid-out array among the inputs or outputs of this call, for diagnostics.
    ///   - `layout`: Layout to shift.
    fn batch_prefixed_layout(&self, role: &str, index: usize, layout: &Layout) -> Result<Layout, BatchingError> {
        match layout {
            Layout::Tiled(layout) => Ok(Layout::Tiled(TiledLayout::new(
                layout.minor_to_major().iter().map(|axis| axis + 1).chain(std::iter::once(0)).collect(),
                layout.tiles().to_vec(),
            ))),
            Layout::Strided(_) => Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "custom call `{}` cannot batch {role} {index} because its strided layout `{layout}` does not \
                     determine the byte stride of the inserted batch axis",
                    self.target_name,
                ),
            }),
        }
    }

    /// Restores explicit aliased layouts after ordinary dense alignment has cleared physical layout metadata.
    fn align_array_alias_layouts<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        inputs: Vec<ArrayBatch<C::Value>>,
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
    where
        C::Operation: From<BroadcastOperation>,
    {
        if self.input_output_aliases.is_empty() {
            return Ok(inputs);
        }
        let batch_dimension = P::axis_dimension(context)?;
        inputs
            .into_iter()
            .enumerate()
            .map(|(input_index, input)| {
                let input_type = input.r#type().into_owned();
                let Some(layout) = self.alias_input_layout(input_index, &input_type, Some(&batch_dimension))? else {
                    return Ok(input);
                };
                if input_type.layout() == Some(&layout) {
                    return Ok(input);
                }
                // An identity broadcast materializes the required storage; changing type metadata alone would leave
                // the kernel reading a buffer with a different physical layout.
                let output_axes = (0..input_type.rank()).collect();
                let operation = BroadcastOperation::new(input_type.with_layout(layout), output_axes);
                let mut outputs = context.parent().bind(operation, Vec::new(), std::slice::from_ref(input.value()))?;
                check_count!("output", outputs, 1, ProgramError);
                ArrayBatch::new(outputs.remove(0), input.batch_axis())?.with_ragged_axes(input.ragged_axes().to_vec())
            })
            .collect()
    }

    /// Restores explicit aliased layouts through dimension-valued identity broadcasts in the mixed universe.
    fn align_array_ir_alias_layouts<C: Context<Type = ArrayIrType>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        inputs: Vec<ArrayIrBatch<C::Value>>,
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError>
    where
        C::Value: DimensionSize,
        C::Operation: From<DynamicBroadcastOperation>,
    {
        if self.input_output_aliases.is_empty() {
            return Ok(inputs);
        }
        let batch_dimension = <&DimensionType>::try_from(context.axis_extent().r#type().as_ref())?.to_dimension();
        inputs
            .into_iter()
            .enumerate()
            .map(|(input_index, input)| {
                let input_type = <&ArrayType>::try_from(input.value().r#type().as_ref())?.clone();
                let Some(layout) = self.alias_input_layout(input_index, &input_type, Some(&batch_dimension))? else {
                    return Ok(input);
                };
                if input_type.layout() == Some(&layout) {
                    return Ok(input);
                }
                let mut values = vec![input.value().clone(), context.axis_extent().clone()];
                for axis in 1..input_type.rank() {
                    values.push(input.value().dimension_size(axis)?);
                }
                let operation =
                    DynamicBroadcastOperation::new((0..input_type.rank()).collect()).with_output_layout(layout);
                let mut outputs = context.parent().bind(operation, Vec::new(), values.as_slice())?;
                check_count!("output", outputs, 1, ProgramError);
                ArrayIrBatch::new(outputs.remove(0), input.batch_axis())?.with_ragged_axes(input.ragged_axes().to_vec())
            })
            .collect()
    }

    /// Returns the lowering-only [`ScanOperation`] unroll factor that the selected sequential [`CustomCallBatching`]
    /// behavior requests for a mapped axis of extent `batch_dimension`, or [`None`] to keep one kernel call per loop
    /// trip. [`SequentialUnrolled`](CustomCallBatching::SequentialUnrolled) uses the static batch extent itself, which
    /// unrolls the scan completely (refer to [`ScanOperation::with_unroll`]), and so it rejects dynamic extents. The
    /// single-call and rejecting behaviors stage no scan and request no unrolling.
    fn sequential_unroll(&self, batch_dimension: &Dimension) -> Result<Option<usize>, BatchingError> {
        match self.batching {
            CustomCallBatching::Sequential { unroll } => Ok(unroll),
            CustomCallBatching::SequentialUnrolled => match batch_dimension.value() {
                Some(extent) => Ok(Some(extent.max(1))),
                None => Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "custom call `{}` batching `{}` requires a statically known batch extent but the mapped axis \
                         has dynamic extent `{batch_dimension}`",
                        self.target_name, self.batching,
                    ),
                }),
            },
            CustomCallBatching::Rejected
            | CustomCallBatching::BroadcastAll
            | CustomCallBatching::ExpandDimensions
            | CustomCallBatching::Vectorized => Ok(None),
        }
    }

    /// Returns the [`BatchingError`] reported when input `index` carries the mapped `batch_axis` and this call has no
    /// way to thread that axis through its opaque kernel.
    fn mapped_input_error(&self, index: usize, batch_axis: BatchAxis) -> BatchingError {
        BatchingError::UnsupportedOperation {
            message: format!(
                "custom call `{}` has no batching rule for input {index} mapped at batch {batch_axis}; invoke a \
                 kernel that understands the batch axis, or select an explicit batching behavior with \
                 `CustomCallOperation::with_batching`",
                self.target_name,
            ),
        }
    }

    /// Returns the forward-mode differentiation of this call in either universe. Foreign kernels have no derivable
    /// derivative, so users must call them through a [`custom_function`](crate::custom_function) with derivative rules
    /// to propagate live tangents (which is also how JAX handles `ffi_call` differentiation). When every input tangent
    /// is a structural zero, including when the call has no inputs, the outputs carry no tangent either, and so the
    /// call is replayed unchanged on the primal inputs and each output is paired with a structural zero tangent. This
    /// case is reached whenever the driver's own zero-tangent shortcut does not apply (e.g., for calls with no inputs).
    fn jvp_without_tangents<C: Context<Operation: From<CustomCallOperation>>>(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
    where
        C::Type: DifferentiableType,
    {
        if inputs.iter().any(|input| !input.tangent().is_zero()) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "custom call `{}` has no differentiation rule; call it through a `custom_function` with \
                     derivative rules to provide one",
                    self.target_name,
                ),
            }
            .into());
        }
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        context
            .bind(self.clone(), Vec::new(), primals.as_slice())?
            .into_iter()
            .map(DifferentiationDual::new_with_zero_tangent)
            .collect()
    }

    /// Returns the declared output types with undeclared shardings inheriting the inputs' manual variation.
    /// Both universes use this rule; dynamic dimensions remain unchanged.
    fn infer_array_output_types(&self, input_types: &[&ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        // Opaque code cannot be assumed to produce the same result on every device, so an output whose declared type
        // says nothing about placement inherits the manual variation of the inputs, placed replicated on their mesh.
        // A declared sharding is the author's statement and is kept as is.
        let Some(input_sharding) = input_types.iter().find_map(|input_type| input_type.sharding()) else {
            return Ok(self.output_types.clone());
        };
        let varying_manual_axes = input_types
            .iter()
            .filter_map(|input_type| input_type.sharding())
            .flat_map(|sharding| sharding.varying_manual_axes().iter().cloned())
            .collect::<BTreeSet<_>>();
        self.output_types
            .iter()
            .map(|output_type| {
                if output_type.sharding().is_some() || varying_manual_axes.is_empty() {
                    return Ok(output_type.clone());
                }
                let sharding = Sharding::replicated(input_sharding.mesh().clone(), output_type.rank())
                    .with_varying_manual_axes(varying_manual_axes.clone())
                    .map_err(|error| TypeError::invalid(error.to_string()))?;
                output_type.clone().with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()))
            })
            .collect()
    }

    /// Validates the attributes, declared input layouts, aliases, and ragged contract of this operation against its
    /// array input types, which both universes share.
    fn validate_configuration(&self, input_types: &[&ArrayType]) -> Result<(), TypeError> {
        if let Some(name) = CustomCallAttribute::duplicate_name(self.attributes.as_slice()) {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` declares attribute `{name}` more than once",
            )));
        }
        self.validate_input_layouts(input_types)?;
        self.validate_input_output_aliases(input_types)?;
        self.validate_ragged_contract(input_types)
    }

    /// Validates the declared input layouts against the array inputs of this operation.
    fn validate_input_layouts(&self, input_types: &[&ArrayType]) -> Result<(), TypeError> {
        if self.input_layouts.is_empty() {
            return Ok(());
        }
        if self.input_layouts.len() != input_types.len() {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` declares {} input layouts but the call has {} array inputs",
                self.input_layouts.len(),
                input_types.len(),
            )));
        }
        for (input_index, (layout, input_type)) in self.input_layouts.iter().zip(input_types).enumerate() {
            let Some(layout) = layout else {
                continue;
            };
            let layout_rank = match layout {
                Layout::Tiled(layout) => layout.rank(),
                Layout::Strided(layout) => layout.rank(),
            };
            if layout_rank != input_type.rank() {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` input layout `{layout}` has rank {layout_rank} but input \
                     {input_index} has type `{input_type}`",
                )));
            }
        }
        Ok(())
    }

    /// Validates flat input/output aliases against the array inputs and declared input layouts of this operation.
    fn validate_input_output_aliases(&self, input_types: &[&ArrayType]) -> Result<(), TypeError> {
        for alias in &self.input_output_aliases {
            let Some(input_type) = input_types.get(alias.input_index) else {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` alias `{alias}` refers to input {} but the call has {} array \
                     inputs",
                    alias.input_index,
                    input_types.len(),
                )));
            };
            let Some(output_type) = self.output_types.get(alias.output_index) else {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` alias `{alias}` refers to output {} but the call has {} outputs",
                    alias.output_index,
                    self.output_types.len(),
                )));
            };
            if *input_type != output_type {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` alias `{alias}` requires matching input and output types but \
                     input {} has type `{}` and output {} has type `{}`",
                    alias.input_index, input_type, alias.output_index, output_type,
                )));
            }
            if let Some(Some(input_layout)) = self.input_layouts.get(alias.input_index)
                && Some(input_layout) != output_type.layout()
            {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` alias `{alias}` requires the declared layout `{input_layout}` of \
                     input {} to match the layout of output {}",
                    alias.input_index, alias.output_index,
                )));
            }
        }
        Ok(())
    }

    /// Validates one declared packed ragged axis and its finite physical capacity.
    fn validate_ragged_axis(
        &self,
        kind: &str,
        index: usize,
        axis: usize,
        r#type: &ArrayType,
        dimension: &DimensionVariable,
    ) -> Result<(), TypeError> {
        let Some(physical_dimension) = r#type.shape().dimensions().get(axis) else {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` ragged contract {kind} {index} axis {axis} is out of bounds for type \
                 `{type}`",
            )));
        };
        let Dimension::Static(physical_extent) = physical_dimension else {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` ragged contract {kind} {index} axis {axis} must have a finite static \
                 physical bound but has dimension `{physical_dimension}`",
            )));
        };
        if dimension.bounds().upper().is_none_or(|upper| upper.saturating_sub(1) > *physical_extent) {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` ragged dimension `{dimension}` with bounds {} exceeds the physical \
                 extent {physical_extent} of {kind} {index} axis {axis}",
                dimension.bounds(),
            )));
        }
        Ok(())
    }

    /// Validates this operation's optional ragged calling convention against its complete array signature.
    fn validate_ragged_contract(&self, input_types: &[&ArrayType]) -> Result<(), TypeError> {
        let Some(contract) = &self.ragged_contract else {
            return Ok(());
        };
        if contract.output_bindings.len() != self.output_types.len() {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` ragged contract declares {} output bindings but the call has {} \
                 outputs",
                contract.output_bindings.len(),
                self.output_types.len(),
            )));
        }
        let expected_extent_rank = contract.batch_prefix_count;
        let expected_extent_type = match expected_extent_rank {
            0 => "an integer scalar".to_string(),
            1 => "a batch-prefixed integer vector".to_string(),
            rank => format!("a rank-{rank} batch-prefixed integer tensor"),
        };

        for (binding_index, binding) in contract.input_bindings.iter().enumerate() {
            if let Some(existing) =
                contract.input_bindings[..binding_index].iter().find(|existing| existing.name == binding.name)
            {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` ragged input binding `{}` duplicates binding `{}`",
                    binding.name, existing.name,
                )));
            }
            if let Some(existing) = contract.input_bindings[..binding_index]
                .iter()
                .find(|existing| existing.input_index == binding.input_index)
            {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` ragged input bindings `{}` and `{}` both bind input {}",
                    existing.name, binding.name, binding.input_index,
                )));
            }
            if let Some(existing) = contract.input_bindings[..binding_index].iter().find(|existing| {
                existing.dimension == binding.dimension && existing.extent_input_index != binding.extent_input_index
            }) {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` ragged input bindings `{}` and `{}` reuse dimension `{}` with \
                     different extent inputs {} and {}",
                    existing.name,
                    binding.name,
                    binding.dimension,
                    existing.extent_input_index,
                    binding.extent_input_index,
                )));
            }
            let Some(input_type) = input_types.get(binding.input_index) else {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` ragged input binding `{}` refers to input {} but the call has \
                     {} array inputs",
                    binding.name,
                    binding.input_index,
                    input_types.len(),
                )));
            };
            self.validate_ragged_axis("input", binding.input_index, binding.axis, input_type, &binding.dimension)?;
            let Some(extent_type) = input_types.get(binding.extent_input_index) else {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` ragged input binding `{}` refers to extent input {} but the \
                     call has {} array inputs",
                    binding.name,
                    binding.extent_input_index,
                    input_types.len(),
                )));
            };
            if extent_type.rank() != expected_extent_rank || !extent_type.data_type().is_integer() {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` ragged input binding `{}` requires extent input {} to be \
                     {expected_extent_type} but got `{extent_type}`",
                    binding.name, binding.extent_input_index,
                )));
            }
        }

        for (output_index, binding) in contract.output_bindings.iter().enumerate() {
            match binding {
                CustomCallRaggedOutputBinding::Preserved { input_binding, axis } => {
                    let Some(input_binding) =
                        contract.input_bindings.iter().find(|binding| binding.name == *input_binding)
                    else {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` ragged output {output_index} preserves unknown input \
                             binding `{input_binding}`",
                        )));
                    };
                    self.validate_ragged_axis(
                        "output",
                        output_index,
                        *axis,
                        &self.output_types[output_index],
                        &input_binding.dimension,
                    )?;
                    if let Some(alias) =
                        self.input_output_aliases.iter().find(|alias| alias.output_index == output_index)
                        && (alias.input_index != input_binding.input_index || *axis != input_binding.axis)
                    {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` alias `{alias}` conflicts with preserved ragged binding \
                             `{}` because aliases require the same packed input, physical axis, dimension identity, \
                             and extent binding",
                            input_binding.name,
                        )));
                    }
                }
                CustomCallRaggedOutputBinding::Consumed => {
                    if let Some(alias) = self.input_output_aliases.iter().find(|alias| {
                        alias.output_index == output_index
                            && contract.input_bindings.iter().any(|binding| binding.input_index == alias.input_index)
                    }) {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` consumed ragged output {output_index} cannot retain alias \
                             `{alias}`",
                        )));
                    }
                }
                CustomCallRaggedOutputBinding::Fresh { axis, extent_output_index, dimension } => {
                    if let Some(input_binding) =
                        contract.input_bindings.iter().find(|binding| binding.dimension == *dimension)
                    {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` fresh ragged output {output_index} dimension \
                             `{dimension}` is already declared by input binding `{}`",
                            input_binding.name,
                        )));
                    }
                    if let Some((existing_output_index, existing_extent_output_index)) =
                        contract.output_bindings[..output_index].iter().enumerate().find_map(
                            |(existing_output_index, binding)| match binding {
                                CustomCallRaggedOutputBinding::Fresh {
                                    extent_output_index: existing_extent_output_index,
                                    dimension: existing_dimension,
                                    ..
                                } if existing_dimension == dimension
                                    && existing_extent_output_index != extent_output_index =>
                                {
                                    Some((existing_output_index, existing_extent_output_index))
                                }
                                _ => None,
                            },
                        )
                    {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` fresh ragged outputs {existing_output_index} and \
                             {output_index} reuse dimension `{dimension}` with different extent outputs \
                             {existing_extent_output_index} and {extent_output_index}",
                        )));
                    }
                    self.validate_ragged_axis(
                        "output",
                        output_index,
                        *axis,
                        &self.output_types[output_index],
                        dimension,
                    )?;
                    let Some(extent_type) = self.output_types.get(*extent_output_index) else {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` fresh ragged output {output_index} refers to extent \
                             output {extent_output_index} but the call has {} outputs",
                            self.output_types.len(),
                        )));
                    };
                    if extent_type.rank() != expected_extent_rank || !extent_type.data_type().is_integer() {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` fresh ragged output {output_index} requires extent output \
                             {extent_output_index} to be {expected_extent_type} but got `{extent_type}`",
                        )));
                    }
                    if let Some(alias) = self.input_output_aliases.iter().find(|alias| {
                        alias.output_index == output_index
                            && contract.input_bindings.iter().any(|binding| binding.input_index == alias.input_index)
                    }) {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` fresh ragged output {output_index} cannot retain alias \
                             `{alias}`",
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Validates homogeneous or mixed ragged input carriers against the declaration and returns active bindings.
    fn active_ragged_bindings<V: Clone + PartialEq>(
        &self,
        contract: &CustomCallRaggedContract,
        inputs: &[CustomCallRaggedInput<'_, V>],
    ) -> Result<Vec<(String, RaggedAxis<V>)>, BatchingError> {
        if contract.ragged_discharged && inputs.iter().any(|input| !input.ragged_axes.is_empty()) {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("custom call `{}` does not support nested ragged batching", self.target_name),
            });
        }

        self.validate_ragged_contract(&inputs.iter().map(|input| &input.physical_type).collect::<Vec<_>>())?;

        let mut active = Vec::new();
        for (input_index, input) in inputs.iter().enumerate() {
            if input.ragged_axes.len() > 1 {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "custom call `{}` supports at most one ragged axis per input but input {input_index} \
                         carries {}",
                        self.target_name,
                        input.ragged_axes.len(),
                    ),
                });
            }
            let Some(ragged_axis) = input.ragged_axes.first() else {
                continue;
            };
            let Some(batch_axis) = input.batch_axis else {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "custom call `{}` cannot discharge ragged input {input_index} without a mapped batch axis",
                        self.target_name,
                    ),
                });
            };
            let logical_axis = ragged_axis.axis() - usize::from(batch_axis < ragged_axis.axis());
            let Some(binding) = contract
                .input_bindings
                .iter()
                .find(|binding| binding.input_index == input_index && binding.axis == logical_axis)
            else {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "custom call `{}` ragged contract does not bind input {input_index} axis {logical_axis}",
                        self.target_name,
                    ),
                });
            };
            if ragged_axis.dimension() != &binding.dimension {
                return Err(BatchingError::InvalidBatchMetadata {
                    message: format!(
                        "custom call `{}` ragged input binding `{}` expects dimension `{}` but input \
                         {input_index} carries `{}`",
                        self.target_name,
                        binding.name,
                        binding.dimension,
                        ragged_axis.dimension(),
                    ),
                });
            }
            let extent_input = &inputs[binding.extent_input_index];
            let Some(extent_batch_axis) = extent_input.batch_axis else {
                return Err(BatchingError::InvalidBatchMetadata {
                    message: format!(
                        "custom call `{}` ragged input binding `{}` requires mapped extent input {}",
                        self.target_name, binding.name, binding.extent_input_index,
                    ),
                });
            };
            let expected_extent_axes = contract.active_extent_axes(batch_axis, extent_batch_axis);
            if ragged_axis.extent_axes() != expected_extent_axes {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "custom call `{}` supports one ragged batching level, but input {input_index} ragged \
                         dimension `{}` has extent-axis mapping `{:?}` instead of `{:?}`",
                        self.target_name,
                        ragged_axis.dimension(),
                        ragged_axis.extent_axes(),
                        expected_extent_axes,
                    ),
                });
            }
            if extent_input.value != ragged_axis.extents() {
                return Err(BatchingError::InvalidBatchMetadata {
                    message: format!(
                        "custom call `{}` ragged input binding `{}` requires input {} to be the exact extent value \
                         carried by input {input_index}",
                        self.target_name, binding.name, binding.extent_input_index,
                    ),
                });
            }
            active.push((binding.name.clone(), ragged_axis.clone()));
        }
        Ok(active)
    }

    /// Rejects ragged input metadata for a call that has no declared ragged contract.
    fn reject_undeclared_ragged_inputs<'o, V: 'o>(
        &self,
        mut inputs: impl Iterator<Item = (usize, &'o [RaggedAxis<V>])>,
    ) -> Result<(), BatchingError> {
        if let Some((index, ragged_axis)) =
            inputs.find_map(|(index, ragged_axes)| ragged_axes.first().map(|axis| (index, axis)))
        {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "custom call `{}` does not support bounded ragged dimension `{}` on input {}",
                    self.target_name,
                    ragged_axis.dimension(),
                    index,
                ),
            });
        }
        Ok(())
    }

    /// Returns the ragged axis that the declared [`CustomCallRaggedContract`] attaches to each output of a rewritten
    /// call, or [`None`] for every output of a call without a contract. Preserved bindings reuse the exact extent input
    /// of their active input binding, and fresh bindings reuse the declared extent output. Inactive preserved bindings
    /// and consumed bindings attach nothing.
    ///
    /// # Parameters
    ///
    ///   - `output_values`: Output values of the rewritten call.
    ///   - `input_value`: Returns the value of the rewritten call's array input at the provided index.
    ///   - `active`: Active ragged input bindings returned by
    ///     [`active_ragged_bindings`](Self::active_ragged_bindings).
    ///   - `batch_prefixed`: Whether the rewritten call's outputs gained a leading batch axis.
    fn ragged_output_axes<V: Clone, F: Fn(usize) -> V>(
        &self,
        output_values: &[V],
        input_value: F,
        active: &[(String, RaggedAxis<V>)],
        batch_prefixed: bool,
    ) -> Vec<Option<RaggedAxis<V>>> {
        let Some(contract) = &self.ragged_contract else {
            return vec![None; output_values.len()];
        };
        let offset = usize::from(batch_prefixed);
        let extent_axes = (0..contract.batch_prefix_count + offset).collect::<Vec<_>>();
        contract
            .output_bindings
            .iter()
            .map(|binding| match binding {
                CustomCallRaggedOutputBinding::Preserved { input_binding, axis } => {
                    active.iter().find(|(name, _)| name == input_binding).map(|(_, source)| {
                        let binding =
                            contract.input_bindings.iter().find(|binding| binding.name == *input_binding).unwrap();
                        RaggedAxis::new(
                            *axis + offset,
                            input_value(binding.extent_input_index),
                            source.dimension().clone(),
                            extent_axes.clone(),
                        )
                    })
                }
                CustomCallRaggedOutputBinding::Consumed => None,
                CustomCallRaggedOutputBinding::Fresh { axis, extent_output_index, dimension } => Some(RaggedAxis::new(
                    *axis + offset,
                    output_values[*extent_output_index].clone(),
                    dimension.clone(),
                    extent_axes.clone(),
                )),
            })
            .collect()
    }

    /// Wraps the outputs of a rewritten homogeneous call as batches on `batch_axis`, attaching the ragged metadata of
    /// [`ragged_output_axes`](Self::ragged_output_axes) and recording the input dimensions that the call consumed.
    fn array_outputs<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>>(
        &self,
        output_values: Vec<C::Value>,
        batch_axis: BatchAxis,
        inputs: &[ArrayBatch<C::Value>],
        active: &[(String, RaggedAxis<C::Value>)],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        let ragged_axes = self.ragged_output_axes(
            output_values.as_slice(),
            |index| inputs[index].value().clone(),
            active,
            !batch_axis.is_replicated(),
        );
        let outputs = output_values
            .into_iter()
            .zip(ragged_axes)
            .map(|(value, ragged_axis)| {
                let output = ArrayBatch::new(value, batch_axis)?;
                match ragged_axis {
                    Some(ragged_axis) => output.with_ragged_axes(vec![ragged_axis]),
                    None => Ok(output),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let consumed = self.ragged_contract.as_ref().map(|contract| contract.consumed_dimensions(active));
        Ok(BatchedOutputs::new(outputs, consumed.unwrap_or_default()))
    }

    /// Wraps the outputs of a rewritten mixed-universe call as batches on `batch_axis`, attaching the ragged metadata
    /// of [`ragged_output_axes`](Self::ragged_output_axes) and recording the input dimensions that the call consumed.
    fn array_ir_outputs<C: Context<Type = ArrayIrType>>(
        &self,
        output_values: Vec<C::Value>,
        batch_axis: BatchAxis,
        inputs: &[ArrayIrBatch<C::Value>],
        active: &[(String, RaggedAxis<C::Value>)],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        let ragged_axes = self.ragged_output_axes(
            output_values.as_slice(),
            |index| inputs[index].value().clone(),
            active,
            !batch_axis.is_replicated(),
        );
        let outputs = output_values
            .into_iter()
            .zip(ragged_axes)
            .map(|(value, ragged_axis)| {
                let output = ArrayIrBatch::new(value, batch_axis)?;
                match ragged_axis {
                    Some(ragged_axis) => output.with_ragged_axes(vec![ragged_axis]),
                    None => Ok(output),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let consumed = self.ragged_contract.as_ref().map(|contract| contract.consumed_dimensions(active));
        Ok(BatchedOutputs::new(outputs, consumed.unwrap_or_default()))
    }

    /// Validates that every ragged input binding of a single-call batching strategy other than
    /// [`BroadcastAll`](CustomCallBatching::BroadcastAll) has mapped data and extent inputs. Those strategies leave
    /// replicated inputs without the full batch extent, which a ragged binding cannot describe.
    fn validate_single_call_ragged_bindings<F: Fn(usize) -> bool>(
        &self,
        is_replicated: F,
    ) -> Result<(), BatchingError> {
        let Some(contract) = &self.ragged_contract else {
            return Ok(());
        };
        if self.batching == CustomCallBatching::BroadcastAll {
            return Ok(());
        }
        match contract
            .input_bindings
            .iter()
            .find(|binding| is_replicated(binding.input_index) || is_replicated(binding.extent_input_index))
        {
            Some(binding) => Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "custom call `{}` batching `{}` requires mapped data and extent inputs for ragged binding `{}`",
                    self.target_name, self.batching, binding.name,
                ),
            }),
            None => Ok(()),
        }
    }
}

impl Display for CustomCallOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for CustomCallOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        CUSTOM_CALL_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let array_input_types = input_types.iter().collect::<Vec<_>>();
        self.validate_configuration(array_input_types.as_slice())?;

        // The homogeneous universe has no way to ground a dynamic result extent: only the mixed form accepts the
        // trailing first-class dimension inputs that define one.
        for output_type in &self.output_types {
            if output_type.static_shape().is_none() {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` requires explicit result-extent inputs for dynamic output type \
                     `{output_type}`",
                )));
            }
        }
        self.infer_array_output_types(array_input_types.as_slice())
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        Cow::Owned(Effects::explicit(self.effect_class.map(EffectClasses::single).unwrap_or(EffectClasses::NONE)))
    }

    #[inline]
    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        self.renamed(renaming)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CUSTOM_CALL_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("target", &self.target_name)?;
            for (name, value) in &self.attributes {
                operation.field(name, value)?;
            }
            if !self.input_layouts.is_empty() {
                let input_layouts = self
                    .input_layouts
                    .iter()
                    .map(|layout| layout.as_ref().map_or_else(|| "default".to_string(), Layout::to_string))
                    .collect::<Vec<_>>();
                operation.field("input_layouts", format_args!("[{}]", input_layouts.join(", ")))?;
            }
            for alias in &self.input_output_aliases {
                operation.field("input_output_alias", alias)?;
            }
            if let Some(effect_class) = self.effect_class {
                operation.field("has_side_effect", true)?;
                if effect_class != EffectClass::OrderedIo {
                    operation.field("effect_class", effect_class)?;
                }
            }
            if self.batching != CustomCallBatching::default() {
                operation.field("batching", self.batching)?;
            }
            if let Some(contract) = &self.ragged_contract {
                operation.field("ragged_contract", contract)?;
            }
            Ok(())
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free CustomCallOperation);

impl<C: Domain<Type = ArrayType, Value: CustomCall>> InterpretableOperation<C> for CustomCallOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        C::Value::custom_call(self, inputs)
    }
}

// Partial evaluation uses the default fold-or-residualize policy. Known calls execute or stage through the parent
// context when supported; pure calls remain residual when an eager parent cannot execute them. Effectful failures
// propagate, and residual calls retain their declared effects through dead-code elimination.
impl<C: Context<Type = ArrayType, Operation: From<CustomCallOperation>>> PartiallyEvaluatableOperation<C>
    for CustomCallOperation
{
}

// Homogeneous-array batching rule for [`CustomCallOperation`]. A foreign kernel is opaque, so Ryft cannot derive how a
// batch axis threads through it. A call whose inputs are *all replicated* is nevertheless bound unchanged through the
// parent context and reports replicated outputs, matching JAX, which only invokes a batching rule once some input is
// actually mapped.
//
// That all-replicated shortcut is sound *for this operation specifically* because a custom call is region-free by
// construction: [`Operation::infer_output_types`] rejects every attached region, so the kernel is a leaf that observes
// only its instruction inputs. A foreign kernel therefore cannot observe the transform's named axis, and running it
// unchanged over replicated inputs computes exactly what each batch item would have computed on its own. The shortcut
// must never be generalized to region-carrying operations, because a region can contain a named-axis operation whose
// value differs per batch item even when every input of the enclosing instruction is replicated.
// For example, mapping a region that returns the named-axis index still produces `[0, 1, 2]` at extent 3 even with
// replicated inputs. Custom-function wrappers therefore always batch their regions structurally.
//
// A mapped input is instead governed by the call's own [`CustomCallBatching`] behavior:
// [`Rejected`](CustomCallBatching::Rejected) reports a [`BatchingError::UnsupportedOperation`] naming that input and
// its mapped axis, [`Sequential`](CustomCallBatching::Sequential) stages one [`ScanOperation`] whose body performs a
// single unbatched call (mapped inputs realigned to batch axis `0` and sliced per iteration, replicated inputs threaded
// as invariant carries), [`SequentialUnrolled`](CustomCallBatching::SequentialUnrolled) stages the same scan with an
// unroll factor equal to the static batch extent, and [`BroadcastAll`](CustomCallBatching::BroadcastAll) aligns every
// input to batch axis `0` and rebinds one call whose declared outputs gain the same leading batch dimension. The other
// single-call modes differ only in whether replicated inputs gain a singleton axis or remain unchanged. All mapped
// behaviors keep the staged program's size independent of the batch extent (unrolling is lowering-only) and compose
// with nested batching, because the rewritten instruction is bound through the parent context and carries the same
// behavior selection.
//
// Both sequential behaviors require a statically known mapped extent: the scan trip count is a host `usize` in this
// universe. The mixed [`ArrayIrType`] rule below owns the dynamic-extent case, where the trip count is a first-class
// dimension input (and which `SequentialUnrolled` rejects, because a dynamic trip count cannot be fully unrolled).
impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for CustomCallOperation
where
    C::Value: PartialEq,
    C::Operation: From<CustomCallOperation> + From<BroadcastOperation> + From<ScanOperation<C::Type>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        let active_ragged_bindings = if let Some(contract) = &self.ragged_contract {
            let ragged_inputs = inputs
                .iter()
                .map(|input| -> Result<_, BatchingError> {
                    Ok(CustomCallRaggedInput {
                        value: input.value(),
                        batch_axis: input.batch_axis_position(),
                        ragged_axes: input.ragged_axes(),
                        physical_type: input.r#type().unbatched(input.batch_axis())?,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.active_ragged_bindings(contract, ragged_inputs.as_slice())?
        } else {
            self.reject_undeclared_ragged_inputs(
                inputs.iter().enumerate().map(|(index, input)| (index, input.ragged_axes())),
            )?;
            Vec::new()
        };
        let Some((index, mapped)) = inputs.iter().enumerate().find(|(_, input)| !input.batch_axis().is_replicated())
        else {
            let values = inputs.iter().map(ArrayBatch::value).cloned().collect::<Vec<_>>();
            let outputs = context.parent().bind(self.clone(), Vec::new(), values.as_slice())?;
            return self.array_outputs::<C, P>(outputs, BatchAxis::replicated(), inputs, &[]);
        };

        match self.batching {
            CustomCallBatching::Rejected => Err(self.mapped_input_error(index, mapped.batch_axis())),
            CustomCallBatching::Sequential { .. } | CustomCallBatching::SequentialUnrolled => {
                // Realign every mapped input to batch axis 0 so the scan consumes one per-item row per iteration, and
                // keep replicated inputs as invariant loop carries.
                let mut carry_indices = Vec::new();
                let mut stacked_indices = Vec::new();
                let mut aligned = Vec::with_capacity(inputs.len());
                for (index, input) in inputs.iter().enumerate() {
                    if input.batch_axis().is_replicated() {
                        carry_indices.push(index);
                        aligned.push(input.clone());
                    } else {
                        stacked_indices.push(index);
                        aligned.push(P::match_axis(context, input, Axis::from(0))?);
                    }
                }

                // Build the scan body: one unbatched application of this same call over
                // `[index, carries..., slices...]`, returning the unchanged carries followed by that item's outputs.
                let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
                builder.add_input(ArrayType::scalar(DataType::I64));
                let mut call_inputs = vec![None; inputs.len()];
                let carry_inputs = carry_indices
                    .iter()
                    .map(|&index| {
                        let input_type = aligned[index].r#type().unbatched(aligned[index].batch_axis())?;
                        let input = builder.add_input(input_type);
                        call_inputs[index] = Some(input);
                        Ok(input)
                    })
                    .collect::<Result<Vec<_>, BatchingError>>()?;
                for &index in &stacked_indices {
                    let input_type = aligned[index].r#type().unbatched(aligned[index].batch_axis())?;
                    call_inputs[index] = Some(builder.add_input(input_type));
                }
                let mut call_inputs = call_inputs.into_iter().map(Option::unwrap).collect::<Vec<_>>();
                for alias in &self.input_output_aliases {
                    let Some(&input) = call_inputs.get(alias.input_index) else {
                        continue;
                    };
                    let input_type = builder.atoms()[input.index()].r#type().into_owned();
                    let Some(layout) = self.alias_input_layout(alias.input_index, &input_type, None)? else {
                        continue;
                    };
                    if input_type.layout() != Some(&layout) {
                        // Scan slices have unspecified storage. Materialize the original alias contract inside the
                        // body, where it can preserve both the input buffer layout and the declared output layout.
                        let output_axes = (0..input_type.rank()).collect();
                        let operation = BroadcastOperation::new(input_type.with_layout(layout), output_axes);
                        call_inputs[alias.input_index] =
                            builder.add_instruction(operation, Vec::new(), vec![input], None)?[0];
                    }
                }
                let ragged_contract = self.ragged_contract.as_ref().map(|contract| {
                    if active_ragged_bindings.is_empty() { contract.clone() } else { contract.ragged_discharged() }
                });
                let operation = Self { ragged_contract, ..self.clone() };
                let outputs = builder.add_instruction(operation, Vec::new(), call_inputs, None)?.to_vec();
                let body_outputs = carry_inputs.iter().copied().chain(outputs).collect::<Vec<_>>();
                let body = builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
                    body_outputs,
                    vec![Placeholder; 1 + inputs.len()],
                    vec![Placeholder; carry_inputs.len() + self.output_types.len()],
                )?;

                let mut scan = ScanOperation::<C::Type>::new(carry_inputs.len(), P::axis_size(context)?);
                if let Some(unroll) = self.sequential_unroll(&P::axis_dimension(context)?)? {
                    scan = scan.with_unroll(unroll)?;
                }
                let packed = carry_indices
                    .iter()
                    .chain(stacked_indices.iter())
                    .map(|&index| aligned[index].value().clone())
                    .collect::<Vec<_>>();
                let mut outputs = context.parent().bind(scan, vec![body], packed.as_slice())?;
                check_count!("output", outputs, carry_inputs.len() + self.output_types.len(), ProgramError);
                outputs.drain(..carry_inputs.len());
                self.array_outputs::<C, P>(outputs, BatchAxis::new(0), aligned.as_slice(), &active_ragged_bindings)
            }
            CustomCallBatching::BroadcastAll
            | CustomCallBatching::ExpandDimensions
            | CustomCallBatching::Vectorized => {
                self.validate_single_call_ragged_bindings(|index| inputs[index].batch_axis().is_replicated())?;
                // Mapped axes always move to physical axis 0. Replicated inputs either gain the full mapped extent,
                // gain a singleton axis for the kernel to broadcast internally, or keep their original geometry.
                let aligned = inputs
                    .iter()
                    .map(|input| {
                        if !input.batch_axis().is_replicated() || self.batching == CustomCallBatching::BroadcastAll {
                            return P::match_axis(context, input, Axis::from(0));
                        }
                        if self.batching == CustomCallBatching::Vectorized {
                            return Ok(input.clone());
                        }
                        let input_type = input.r#type();
                        let output_type = input_type.batched(0, Dimension::Static(1), ShardingDimension::Replicated)?;
                        let output_axes = (1..=input_type.rank()).collect();
                        let dimension_sources = std::iter::once(DimensionSource::Static(1))
                            .chain(input_type.shape().dimensions().iter().enumerate().map(|(axis, dimension)| {
                                match dimension {
                                    Dimension::Static(extent) => DimensionSource::Static(*extent),
                                    Dimension::Dynamic(_) => {
                                        DimensionSource::Value { source: input.value().clone(), axis }
                                    }
                                }
                            }))
                            .collect();
                        P::broadcast_input(context, input, output_type, output_axes, Axis::from(0), dimension_sources)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let aligned = self.align_array_alias_layouts::<C, P>(context, aligned)?;
                let aligned_types = aligned.iter().map(|batch| batch.r#type().into_owned()).collect::<Vec<_>>();
                let output_types = self.batch_prefixed_output_types(
                    aligned_types.as_slice(),
                    P::axis_dimension(context)?,
                    context.axis_sharding(),
                )?;
                let values = aligned.iter().map(ArrayBatch::value).cloned().collect::<Vec<_>>();
                let ragged_contract = self
                    .ragged_contract
                    .as_ref()
                    .map(|contract| contract.batch_prefixed(!active_ragged_bindings.is_empty()));
                let input_layouts = self.batch_prefixed_input_layouts(|index| {
                    self.batching != CustomCallBatching::Vectorized || !inputs[index].batch_axis().is_replicated()
                })?;
                let operation = Self { output_types, input_layouts, ragged_contract, ..self.clone() };
                let outputs = context.parent().bind(operation, Vec::new(), values.as_slice())?;
                self.array_outputs::<C, P>(outputs, BatchAxis::new(0), aligned.as_slice(), &active_ragged_bindings)
            }
        }
    }
}

impl_differentiable_operation! {
    CustomCallOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType, Operation: From<CustomCallOperation>>,
    {
        |operation, context, _driver, inputs| {
            operation.jvp_without_tangents(context.primal(), inputs)
        }
    },
    transpose = @nonlinear,
}

// In the mixed array/dimension universe the call additionally consumes one trailing first-class dimension input per
// dynamic axis occurrence across its declared outputs (ordered by output, then by axis), and type inference verifies
// that each input defines exactly the variable referenced by its output axis. The trailing inputs never reach the
// foreign kernel; they only ground the declared logical result extents.
impl MemberOperation<ArrayIrType> for CustomCallOperation {
    fn infer_parent_region_input_types(
        &self,
        _input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        Ok(Vec::new())
    }

    fn infer_parent_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let dynamic_output_dimensions = self
            .output_types
            .iter()
            .flat_map(|output_type| output_type.shape().dimensions())
            .filter_map(Dimension::variable)
            .collect::<Vec<_>>();
        let Some(array_input_count) = input_types.len().checked_sub(dynamic_output_dimensions.len()) else {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` expects {} trailing output-extent dimensions but only {} inputs were \
                 provided",
                dynamic_output_dimensions.len(),
                input_types.len(),
            )));
        };
        let array_input_types =
            input_types[..array_input_count].iter().map(<&ArrayType>::try_from).collect::<Result<Vec<_>, _>>()?;
        self.validate_configuration(array_input_types.as_slice())?;
        for (input_type, expected_variable) in input_types[array_input_count..].iter().zip(dynamic_output_dimensions) {
            let actual_variable = <&DimensionType>::try_from(input_type)?.variable();
            if actual_variable != expected_variable {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` output-extent input defines dimension variable \
                     `{actual_variable}`, but the corresponding declared output axis refers to \
                     `{expected_variable}`",
                )));
            }
        }
        Ok(self.infer_array_output_types(array_input_types.as_slice())?.into_iter().map(Into::into).collect())
    }

    #[inline]
    fn rename_parent_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<DimensionVariable>,
    ) -> Result<Self, TypeError> {
        self.renamed(renaming)
    }
}

// Mixed-universe interpretation splits the inputs into the kernel's array inputs and the trailing declared result
// extents, runs the kernel through the projected array value family, and verifies that every dynamic output axis has
// exactly the extent its input declared before lifting the outputs back into the parent value family.
impl<C> MemberInterpretableOperation<C> for CustomCallOperation
where
    C: Domain<
            Type = ArrayIrType,
            Value: ValueProjection<
                ArrayType,
                Projected: Value<Type = ArrayType> + CustomCall + DimensionSize<usize>,
            > + ValueProjection<DimensionType, Projected = DimensionValue>,
        >,
{
    fn interpret_in_parent<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_parent_output_types(input_types.as_slice(), &[])?;
        let array_input_count = inputs.len() - self.dynamic_output_dimension_count();
        let array_inputs = inputs[..array_input_count]
            .iter()
            .cloned()
            .map(<C::Value as ValueProjection<ArrayType>>::into_projected)
            .collect::<Result<Vec<_>, _>>()?;
        let output_extents = inputs[array_input_count..]
            .iter()
            .cloned()
            .map(<C::Value as ValueProjection<DimensionType>>::into_projected)
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = <C::Value as ValueProjection<ArrayType>>::Projected::custom_call(self, array_inputs.iter())?;
        check_count!("output", outputs, self.output_types.len(), ProgramError);
        let mut output_extents = output_extents.into_iter();
        for (output_index, (output, output_type)) in outputs.iter().zip(&self.output_types).enumerate() {
            for (axis, dimension) in output_type.shape().dimensions().iter().enumerate() {
                if matches!(dimension, Dimension::Dynamic(_)) {
                    let expected_extent = output_extents.next().unwrap().extent();
                    let actual_extent = output.dimension_size(axis)?;
                    if actual_extent != expected_extent {
                        return Err(ProgramError::InvalidArgument {
                            message: format!(
                                "`{CUSTOM_CALL_OPERATION_NAME}` output {output_index} axis {axis} has extent \
                                 {actual_extent}, but its explicit extent input is {expected_extent}",
                            ),
                        });
                    }
                }
            }
        }
        Ok(outputs.into_iter().map(<C::Value as ValueProjection<ArrayType>>::from_projected).collect())
    }
}

// Mixed array/dimension batching rule for [`CustomCallOperation`]. It applies the same all-replicated shortcut and
// the same [`CustomCallBatching`] behaviors as the homogeneous rule above, with two composite-universe additions.
//
// Every trailing first-class output-extent input must be replicated: a per-batch-item extent would make the call's
// results ragged, requiring packed-buffer contracts rather than first-class dimension inputs. An extent-free call
// whose mapped extent is statically known is exactly the homogeneous contract, so it delegates to the projected
// homogeneous rule through [`batch_projected_operation`]. A dynamic mapped extent stays here, because a first-class
// dimension is not an array value and cannot cross the projected array boundary as a scan trip count or broadcast
// extent.
//
// [`Sequential`](CustomCallBatching::Sequential) and [`SequentialUnrolled`](CustomCallBatching::SequentialUnrolled)
// thread the replicated extents as leading invariant scan carries and consume the mapped rows one per iteration, so the
// body's call sees exactly the per-item extents it declared.
// [`BroadcastAll`](CustomCallBatching::BroadcastAll) instead rebinds one call whose declared outputs gain the mapped
// batch dimension, prepending the transform's extent value to each output's trailing extent group when that batch
// dimension is itself dynamic.
impl<C: Context<Type = ArrayIrType>> MemberBatchableOperation<C, ArrayIrBatchingPolicy> for CustomCallOperation
where
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: PartialEq
        + DimensionSize
        + DynamicBroadcast
        + ValueProjection<ArrayType, Projected: PartialEq + Value<Type = ArrayType> + Transpose>,
    C::Operation: From<CustomCallOperation>
        + From<DynamicBroadcastOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + From<ScanOperation<C::Type>>
        + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<CustomCallOperation>
        + From<BroadcastOperation>
        + From<ScanOperation<ArrayType>>
        + From<TransposeOperation>,
{
    fn batch_in_parent<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        let extent_count = self.dynamic_output_dimension_count();
        let Some(array_input_count) = inputs.len().checked_sub(extent_count) else {
            return Err(ProgramError::InvalidInputCount { expected: extent_count, actual: inputs.len() }.into());
        };
        let (arrays, extents) = inputs.split_at(array_input_count);
        let active_ragged_bindings = if let Some(contract) = &self.ragged_contract {
            let ragged_inputs = arrays
                .iter()
                .map(|input| -> Result<_, BatchingError> {
                    let value_type = input.value().r#type();
                    Ok(CustomCallRaggedInput {
                        value: input.value(),
                        batch_axis: input.batch_axis_position(),
                        ragged_axes: input.ragged_axes(),
                        physical_type: <&ArrayType>::try_from(value_type.as_ref())?.unbatched(input.batch_axis())?,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.active_ragged_bindings(contract, ragged_inputs.as_slice())?
        } else {
            self.reject_undeclared_ragged_inputs(
                arrays.iter().enumerate().map(|(index, input)| (index, input.ragged_axes())),
            )?;
            Vec::new()
        };
        for extent in extents {
            extent.validate_replicated_dimension()?;
        }
        let batch_dimension = <&DimensionType>::try_from(context.axis_extent().r#type().as_ref())?.to_dimension();
        if extents.is_empty() && batch_dimension.value().is_some() {
            return batch_projected_operation(context, self, inputs);
        }

        let Some((index, mapped)) = arrays.iter().enumerate().find(|(_, input)| !input.batch_axis().is_replicated())
        else {
            let values = inputs.iter().map(ArrayIrBatch::value).cloned().collect::<Vec<_>>();
            let outputs = context.parent().bind(self.clone(), Vec::new(), values.as_slice())?;
            return self.array_ir_outputs::<C>(outputs, BatchAxis::replicated(), inputs, &[]);
        };

        match self.batching {
            CustomCallBatching::Rejected => Err(self.mapped_input_error(index, mapped.batch_axis())),
            CustomCallBatching::Sequential { .. } | CustomCallBatching::SequentialUnrolled => {
                let mut carry_indices = Vec::new();
                let mut stacked_indices = Vec::new();
                let mut aligned = Vec::with_capacity(arrays.len());
                for (index, input) in arrays.iter().enumerate() {
                    if input.batch_axis().is_replicated() {
                        carry_indices.push(index);
                        aligned.push(input.clone());
                    } else {
                        stacked_indices.push(index);
                        aligned.push(driver.align_batch_axis(context, input.clone(), Axis::from(0))?);
                    }
                }

                // The replicated extents lead the carries so the body's call can reuse them verbatim as its own
                // trailing extent inputs, exactly as the unbatched call declared them.
                let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
                builder.add_input(ArrayType::scalar(DataType::I64).into());
                let mut carry_inputs =
                    extents.iter().map(|extent| builder.add_input(extent.unbatched_type().clone())).collect::<Vec<_>>();
                let mut call_inputs = vec![None; arrays.len()];
                for &index in &carry_indices {
                    let value_type = aligned[index].value().r#type();
                    let input_type =
                        <&ArrayType>::try_from(value_type.as_ref())?.unbatched(aligned[index].batch_axis())?;
                    let input = builder.add_input(input_type.into());
                    carry_inputs.push(input);
                    call_inputs[index] = Some(input);
                }
                for &index in &stacked_indices {
                    let value_type = aligned[index].value().r#type();
                    let input_type =
                        <&ArrayType>::try_from(value_type.as_ref())?.unbatched(aligned[index].batch_axis())?;
                    call_inputs[index] = Some(builder.add_input(input_type.into()));
                }
                let mut call_inputs = call_inputs
                    .into_iter()
                    .map(Option::unwrap)
                    .chain(carry_inputs[..extents.len()].iter().copied())
                    .collect::<Vec<_>>();
                for alias in &self.input_output_aliases {
                    let Some(&input) = call_inputs.get(alias.input_index) else {
                        continue;
                    };
                    let input_type = <&ArrayType>::try_from(builder.atoms()[input.index()].r#type().as_ref())?.clone();
                    let Some(layout) = self.alias_input_layout(alias.input_index, &input_type, None)? else {
                        continue;
                    };
                    if input_type.layout() != Some(&layout) {
                        // Resolve dimensions from the slice itself so a relayout preserves dynamic geometry exactly.
                        let mut broadcast_inputs = vec![input];
                        for axis in 0..input_type.rank() {
                            let dimension = DimensionSizeOperation::new(&input_type, axis)?;
                            let extent = builder.add_instruction(dimension, Vec::new(), vec![input], None)?[0];
                            broadcast_inputs.push(extent);
                        }
                        let operation =
                            DynamicBroadcastOperation::new((0..input_type.rank()).collect()).with_output_layout(layout);
                        call_inputs[alias.input_index] =
                            builder.add_instruction(operation, Vec::new(), broadcast_inputs, None)?[0];
                    }
                }
                let ragged_contract = self.ragged_contract.as_ref().map(|contract| {
                    if active_ragged_bindings.is_empty() { contract.clone() } else { contract.ragged_discharged() }
                });
                let operation = Self { ragged_contract, ..self.clone() };
                let outputs = builder.add_instruction(operation, Vec::new(), call_inputs, None)?.to_vec();
                let body_outputs = carry_inputs.iter().copied().chain(outputs).collect::<Vec<_>>();
                let body = builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
                    body_outputs,
                    vec![Placeholder; 1 + carry_inputs.len() + stacked_indices.len()],
                    vec![Placeholder; carry_inputs.len() + self.output_types.len()],
                )?;

                let mut scan = ScanOperation::<C::Type>::new(carry_inputs.len(), batch_dimension.clone());
                if let Some(unroll) = self.sequential_unroll(&batch_dimension)? {
                    scan = scan.with_unroll(unroll)?;
                }
                let mut packed = extents.iter().map(|extent| extent.value().clone()).collect::<Vec<_>>();
                packed.extend(
                    carry_indices.iter().chain(stacked_indices.iter()).map(|&index| aligned[index].value().clone()),
                );
                if batch_dimension.variable().is_some() {
                    packed.push(context.axis_extent().clone());
                }
                let mut outputs = context.parent().bind(scan, vec![body], packed.as_slice())?;
                check_count!("output", outputs, carry_inputs.len() + self.output_types.len(), ProgramError);
                outputs.drain(..carry_inputs.len());
                self.array_ir_outputs::<C>(outputs, BatchAxis::new(0), aligned.as_slice(), &active_ragged_bindings)
            }
            CustomCallBatching::BroadcastAll
            | CustomCallBatching::ExpandDimensions
            | CustomCallBatching::Vectorized => {
                self.validate_single_call_ragged_bindings(|index| arrays[index].batch_axis().is_replicated())?;
                // First-class output extents stay separate from the kernel's array inputs. Singleton expansion
                // preserves each invariant array's own extents and placement rather than using the mapped extent.
                let aligned = arrays
                    .iter()
                    .map(|input| {
                        if !input.batch_axis().is_replicated() || self.batching == CustomCallBatching::BroadcastAll {
                            return driver.align_batch_axis(context, input.clone(), Axis::from(0));
                        }
                        if self.batching == CustomCallBatching::Vectorized {
                            return Ok(input.clone());
                        }
                        let value_type = input.value().r#type();
                        let input_type = <&ArrayType>::try_from(value_type.as_ref())?;
                        let output_type = input_type.batched(0, Dimension::Static(1), ShardingDimension::Replicated)?;
                        let mut output_dimensions = context.parent().bind(
                            ConstantOperation::new(DimensionValue::constant(1).unwrap()),
                            Vec::new(),
                            &[],
                        )?;
                        for axis in 0..input_type.rank() {
                            output_dimensions.push(input.value().dimension_size(axis)?);
                        }
                        let output_axes = (1..=input_type.rank()).collect::<Vec<_>>();
                        let value = input.value().dynamic_broadcast_with_output_sharding(
                            &output_dimensions,
                            &output_axes,
                            output_type.sharding().cloned(),
                        )?;
                        ArrayIrBatch::new(value, BatchAxis::new(0))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let aligned = self.align_array_ir_alias_layouts::<C>(context, aligned)?;
                let aligned_types = aligned
                    .iter()
                    .map(|batch| Ok(<&ArrayType>::try_from(batch.value().r#type().as_ref())?.clone()))
                    .collect::<Result<Vec<_>, TypeError>>()?;
                let output_types = self.batch_prefixed_output_types(
                    aligned_types.as_slice(),
                    batch_dimension.clone(),
                    context.axis_sharding(),
                )?;
                let ragged_contract = self
                    .ragged_contract
                    .as_ref()
                    .map(|contract| contract.batch_prefixed(!active_ragged_bindings.is_empty()));
                let input_layouts = self.batch_prefixed_input_layouts(|index| {
                    self.batching != CustomCallBatching::Vectorized || !arrays[index].batch_axis().is_replicated()
                })?;
                let operation = Self { output_types, input_layouts, ragged_contract, ..self.clone() };

                // Regroup the trailing extents: each output's inserted batch axis is its new leading dynamic axis,
                // followed by that output's originally declared extents in axis order.
                let mut values = aligned.iter().map(|batch| batch.value().clone()).collect::<Vec<_>>();
                let mut declared_extents = extents.iter();
                for output_type in &self.output_types {
                    if batch_dimension.variable().is_some() {
                        values.push(context.axis_extent().clone());
                    }
                    for dimension in output_type.shape().dimensions() {
                        if matches!(dimension, Dimension::Dynamic(_)) {
                            values.push(declared_extents.next().unwrap().value().clone());
                        }
                    }
                }
                let outputs = context.parent().bind(operation, Vec::new(), values.as_slice())?;
                self.array_ir_outputs::<C>(outputs, BatchAxis::new(0), aligned.as_slice(), &active_ragged_bindings)
            }
        }
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<CustomCallOperation>>> MemberDifferentiableOperation<C>
    for CustomCallOperation
{
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The trailing output-extent inputs are integer dimensions whose tangents are always structural zeros, so the
        // mixed universe follows the same rule as the homogeneous one.
        self.jvp_without_tangents(context.primal(), inputs)
    }
}

/// Represents the ability to call foreign kernels registered with the executing backend. [`CustomCall`] stages or
/// executes a [`CustomCallOperation`]; refer to its documentation for the calling convention and transform rules.
/// Context-carrying values dispatch through the first input's context, so zero-input calls in those contexts must be
/// staged through a [`ProgramBuilder`]. Concrete eager values delegate to their backend, which determines whether
/// zero-input kernels are supported.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
#[capability]
pub trait CustomCall<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Calls the foreign kernel described by `operation` with the provided inputs, returning one value per
    /// declared output type, and a [`ProgramError`] if something goes wrong.
    fn custom_call<'o, I: IntoIterator<Item = &'o Self>>(
        operation: &CustomCallOperation,
        inputs: I,
    ) -> Result<Vec<Self>, ProgramError>
    where
        Self: 'o;
}

impl CustomCall for Array {
    fn custom_call<'o, I: IntoIterator<Item = &'o Self>>(
        operation: &CustomCallOperation,
        _inputs: I,
    ) -> Result<Vec<Self>, ProgramError> {
        // The reference backend has no foreign-kernel registry.
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "the reference array backend cannot execute the foreign kernel `{}`",
                operation.target_name(),
            ),
        })
    }
}

impl<A: Value<Type = ArrayType> + CustomCall + DimensionSize<usize>> CustomCall<ArrayIrType> for ArrayIrValue<A> {
    fn custom_call<'o, I: IntoIterator<Item = &'o Self>>(
        operation: &CustomCallOperation,
        inputs: I,
    ) -> Result<Vec<Self>, ProgramError>
    where
        Self: 'o,
    {
        // A concrete composite value executes the kernel on its array members through the mixed-universe interpretation
        // rule, which also verifies the extents of dynamic outputs against the trailing dimension inputs. Its eager
        // dispatch domain binds only constants, so the generic implementation below never applies to it.
        let inputs = inputs.into_iter().cloned().collect::<Vec<_>>();
        operation.interpret_in_parent(&EagerContext::<Self>::new(), &EmptyRegionDriver, inputs.as_slice())
    }
}

impl<T: Type, V: Value<Type = T>> CustomCall<T> for V
where
    V::DispatchDomain: Context<Operation: From<CustomCallOperation>>,
{
    fn custom_call<'o, I: IntoIterator<Item = &'o Self>>(
        operation: &CustomCallOperation,
        inputs: I,
    ) -> Result<Vec<Self>, ProgramError>
    where
        Self: 'o,
    {
        // Any context-carrying value calls foreign kernels by binding a `CustomCallOperation` through its own context,
        // in either universe, because the operation is also a native member operation of the composite array IR
        // universe. The `From<CustomCallOperation>` bound makes this disjoint from the eager value types (whose context
        // operation is `ConstantOperation`), so it covers the transform tracers and backend-owned values without
        // conflicting with the concrete implementations above.
        let inputs = inputs.into_iter().cloned().collect::<Vec<_>>();
        let Some(first) = inputs.first() else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "the custom-call capability dispatches through its first input's context, so calling `{}` with \
                     no inputs requires staging the operation through a program builder instead",
                    operation.target_name(),
                ),
            });
        };
        first.dispatch_domain().bind(operation.clone(), Vec::new(), inputs.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue,
        ArrayOperation, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue, DimensionVariable,
        LogicalMesh, MeshAxis, MeshAxisType, RaggedAxis, Shape, ShardingDimension, StridedLayout, Tile, TileDimension,
    };
    use crate::batching::{
        BatchAxis, BatchableOperation, BatchedProgram, BatchingContext, BatchingError, BatchingTracer,
        ProgramBatchingOutputAxesPolicy,
    };
    use crate::contexts::StagingContext;
    use crate::differentiation::DifferentiableOperation;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{check_operation_transposition, check_operation_type_inference};
    use crate::parameters::{Parameter, Placeholder};
    use crate::partial::PartialValue;
    use crate::programs::{BindingRegionDriver, MaybeZero, ProgramBuilder, Provenance, ProvenanceScope};
    use crate::tests::hash_of;
    use crate::tracing::{DomainTracer, Trace, TracingContext};

    use super::*;

    /// Returns a one-input preserved-output ragged contract over a packed `f32[4]` value and scalar extent input.
    fn preserved_ragged_contract(dimension: DimensionVariable) -> CustomCallRaggedContract {
        CustomCallRaggedContract::new(
            vec![CustomCallRaggedInputBinding::new("data", 0, 0, 1, dimension)],
            vec![CustomCallRaggedOutputBinding::Preserved { input_binding: "data".to_string(), axis: 0 }],
        )
    }

    /// Batches `operation` in the mixed [`ArrayIrType`] universe over a dynamic `batch ∈ [1, 9)` extent, with a mapped
    /// `f32[batch, 2, 3]` first input and a replicated `f32[2, 3]` second input, and returns the rendering of the
    /// staged program. The dynamic extent keeps the call on the mixed batching rule instead of the projected
    /// homogeneous one.
    fn batch_array_ir_custom_call(operation: CustomCallOperation) -> Result<String, ProgramError> {
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::from(batch.clone()).into());
        let mapped = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2), Dimension::Static(3)]),
            )
            .into(),
        );
        let replicated = trace.input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(replicated)),
        ];
        let outputs = context
            .bind(operation, Vec::new(), &inputs)?
            .into_iter()
            .map(|output| output.into_batch().into_value().atom_id())
            .collect::<Result<Vec<_>, _>>()?;
        let output_count = outputs.len();
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            outputs,
            vec![Placeholder; 3],
            vec![Placeholder; output_count],
        )?;
        Ok(program.to_string())
    }

    #[test]
    fn test_custom_call_attribute() {
        // Every scalar conversion selects the variant of its Rust type, and unsuffixed literals follow Rust's default
        // literal types.
        assert_eq!(CustomCallAttribute::from("x"), CustomCallAttribute::String("x".to_string()));
        assert_eq!(CustomCallAttribute::from("x".to_string()), CustomCallAttribute::String("x".to_string()));
        assert_eq!(CustomCallAttribute::from(true), CustomCallAttribute::Boolean(true));
        assert_eq!(CustomCallAttribute::from(-1i8), CustomCallAttribute::I8(-1));
        assert_eq!(CustomCallAttribute::from(-2i16), CustomCallAttribute::I16(-2));
        assert_eq!(CustomCallAttribute::from(-3i32), CustomCallAttribute::I32(-3));
        assert_eq!(CustomCallAttribute::from(-4i64), CustomCallAttribute::I64(-4));
        assert_eq!(CustomCallAttribute::from(5u8), CustomCallAttribute::U8(5));
        assert_eq!(CustomCallAttribute::from(6u16), CustomCallAttribute::U16(6));
        assert_eq!(CustomCallAttribute::from(7u32), CustomCallAttribute::U32(7));
        assert_eq!(CustomCallAttribute::from(8u64), CustomCallAttribute::U64(8));
        assert_eq!(CustomCallAttribute::from(2.5f32), CustomCallAttribute::F32(2.5));
        assert_eq!(CustomCallAttribute::from(2.5f64), CustomCallAttribute::F64(2.5));
        assert_eq!(CustomCallAttribute::from(4), CustomCallAttribute::I32(4));
        assert_eq!(CustomCallAttribute::from(2.5), CustomCallAttribute::F64(2.5));

        // Every numeric vector except `Vec<u8>` (which is binary data) becomes an array attribute.
        assert_eq!(
            CustomCallAttribute::from(vec![1i8, 2]),
            CustomCallAttribute::Array(CustomCallArrayAttribute::I8(vec![1, 2])),
        );
        assert_eq!(
            CustomCallAttribute::from(vec![1i16, 2]),
            CustomCallAttribute::Array(CustomCallArrayAttribute::I16(vec![1, 2])),
        );
        assert_eq!(
            CustomCallAttribute::from(vec![1i32, 2]),
            CustomCallAttribute::Array(CustomCallArrayAttribute::I32(vec![1, 2])),
        );
        assert_eq!(
            CustomCallAttribute::from(vec![1i64, 2]),
            CustomCallAttribute::Array(CustomCallArrayAttribute::I64(vec![1, 2])),
        );
        assert_eq!(
            CustomCallAttribute::from(vec![1u16, 2]),
            CustomCallAttribute::Array(CustomCallArrayAttribute::U16(vec![1, 2])),
        );
        assert_eq!(
            CustomCallAttribute::from(vec![1u32, 2]),
            CustomCallAttribute::Array(CustomCallArrayAttribute::U32(vec![1, 2])),
        );
        assert_eq!(
            CustomCallAttribute::from(vec![1u64, 2]),
            CustomCallAttribute::Array(CustomCallArrayAttribute::U64(vec![1, 2])),
        );
        assert_eq!(
            CustomCallAttribute::from(vec![1.5f32, 2.0]),
            CustomCallAttribute::Array(CustomCallArrayAttribute::F32(vec![1.5, 2.0])),
        );
        assert_eq!(
            CustomCallAttribute::from(vec![1.5f64, 2.0]),
            CustomCallAttribute::Array(CustomCallArrayAttribute::F64(vec![1.5, 2.0])),
        );
        assert_eq!(
            CustomCallAttribute::from(CustomCallArrayAttribute::U8(vec![1, 2])),
            CustomCallAttribute::Array(CustomCallArrayAttribute::U8(vec![1, 2])),
        );

        // `I64` and `F64` render as bare literals, while every other scalar carries its Rust type suffix.
        let dictionary = CustomCallAttribute::Dictionary(vec![
            ("count".to_string(), CustomCallAttribute::I64(1)),
            ("options".to_string(), CustomCallAttribute::Dictionary(vec![("mode".to_string(), "fast".into())])),
        ]);
        for (attribute, display, debug) in [
            (CustomCallAttribute::String("x".to_string()), "x", "String(\"x\")"),
            (CustomCallAttribute::Bytes(vec![0, 255]), "bytes [00, ff]", "Bytes([0, 255])"),
            (CustomCallAttribute::Boolean(true), "true", "Boolean(true)"),
            (CustomCallAttribute::I8(-1), "-1i8", "I8(-1)"),
            (CustomCallAttribute::I16(-2), "-2i16", "I16(-2)"),
            (CustomCallAttribute::I32(-3), "-3i32", "I32(-3)"),
            (CustomCallAttribute::I64(-4), "-4", "I64(-4)"),
            (CustomCallAttribute::U8(5), "5u8", "U8(5)"),
            (CustomCallAttribute::U16(6), "6u16", "U16(6)"),
            (CustomCallAttribute::U32(7), "7u32", "U32(7)"),
            (CustomCallAttribute::U64(8), "8u64", "U64(8)"),
            (CustomCallAttribute::F32(1.0), "1.0f32", "F32(1.0)"),
            (CustomCallAttribute::F64(2.0), "2.0", "F64(2.0)"),
            (CustomCallAttribute::from(vec![1i32, 2]), "array<i32: 1, 2>", "Array(I32([1, 2]))"),
            (CustomCallAttribute::Dictionary(Vec::new()), "{}", "Dictionary([])"),
            (
                dictionary,
                "{count=1, options={mode=fast}}",
                "Dictionary([(\"count\", I64(1)), (\"options\", Dictionary([(\"mode\", String(\"fast\"))]))])",
            ),
        ] {
            assert_eq!(attribute.to_string(), display);
            assert_eq!(format!("{attribute:?}"), debug);
        }
    }

    #[test]
    fn test_custom_call_attribute_bytes() {
        let bytes = vec![0, 0x80, 0xff];
        let attribute = CustomCallAttribute::from(bytes.clone());
        assert_eq!(attribute, CustomCallAttribute::Bytes(bytes.clone()));
        assert_eq!(CustomCallAttribute::from(bytes.as_slice()), attribute);
        assert_eq!(attribute.to_string(), "bytes [00, 80, ff]");
        assert_eq!(format!("{attribute:?}"), "Bytes([0, 128, 255])");
        assert_eq!(CustomCallAttribute::Bytes(vec![]).to_string(), "bytes []");
        assert_ne!(attribute, CustomCallAttribute::String("bytes [00, 80, ff]".to_string()));
        assert_eq!(
            CustomCallOperation::new("binary", vec![]).with_attribute("payload", bytes.clone()).attributes(),
            &[("payload".to_string(), CustomCallAttribute::Bytes(bytes))],
        );
    }

    #[test]
    fn test_custom_call_attribute_equality() {
        // Equal attributes are equal and hash identically, so they work as map keys.
        let attribute = CustomCallAttribute::F64(2.5);
        assert_eq!(attribute, attribute.clone());
        assert_eq!(hash_of(&attribute), hash_of(&CustomCallAttribute::from(2.5)));
        assert_eq!(hash_of(&CustomCallAttribute::from("name")), hash_of(&CustomCallAttribute::from("name")));
        let attributes = HashMap::from([
            (attribute, "scale"),
            (CustomCallAttribute::F32(2.5), "narrow_scale"),
            (CustomCallAttribute::from(vec![1i32, 2]), "sizes"),
            (CustomCallAttribute::Dictionary(vec![("mode".to_string(), "fast".into())]), "options"),
        ]);
        assert_eq!(attributes.get(&CustomCallAttribute::F64(2.5)), Some(&"scale"));
        assert_eq!(attributes.get(&CustomCallAttribute::F32(2.5)), Some(&"narrow_scale"));
        assert_eq!(attributes.get(&CustomCallAttribute::from(vec![1i32, 2])), Some(&"sizes"));
        assert_eq!(
            attributes.get(&CustomCallAttribute::Dictionary(vec![("mode".to_string(), "fast".into())])),
            Some(&"options"),
        );
        assert_eq!(attributes.get(&CustomCallAttribute::F64(3.5)), None);

        // Floating-point attributes compare bitwise: every NaN equals itself, and signed zeros are distinct.
        let nan = CustomCallAttribute::F64(f64::NAN);
        assert_eq!(nan, nan.clone());
        assert_eq!(hash_of(&nan), hash_of(&CustomCallAttribute::F64(f64::NAN)));
        assert_ne!(nan, CustomCallAttribute::F64(-f64::NAN));
        assert_ne!(CustomCallAttribute::F64(-0.0), CustomCallAttribute::F64(0.0));
        assert_eq!(CustomCallAttribute::F32(f32::NAN), CustomCallAttribute::F32(f32::NAN));
        assert_eq!(hash_of(&CustomCallAttribute::F32(f32::NAN)), hash_of(&CustomCallAttribute::F32(f32::NAN)));
        assert_ne!(CustomCallAttribute::F32(-0.0), CustomCallAttribute::F32(0.0));

        // Attributes of different variants never compare equal, even when they render identically or hold equal
        // values of different widths.
        assert_ne!(CustomCallAttribute::I64(1), CustomCallAttribute::F64(1.0));
        assert_ne!(CustomCallAttribute::I64(1), CustomCallAttribute::Boolean(true));
        assert_ne!(CustomCallAttribute::I32(1), CustomCallAttribute::I64(1));
        assert_ne!(CustomCallAttribute::U8(1), CustomCallAttribute::I8(1));
        assert_ne!(CustomCallAttribute::F32(1.0), CustomCallAttribute::F64(1.0));
        assert_ne!(CustomCallAttribute::String("true".to_string()), CustomCallAttribute::Boolean(true));
        assert_ne!(CustomCallAttribute::String(String::new()), CustomCallAttribute::Bytes(vec![]));

        // Dictionaries compare their entries in insertion order.
        assert_ne!(
            CustomCallAttribute::Dictionary(vec![
                ("a".to_string(), CustomCallAttribute::I64(1)),
                ("b".to_string(), CustomCallAttribute::I64(2)),
            ]),
            CustomCallAttribute::Dictionary(vec![
                ("b".to_string(), CustomCallAttribute::I64(2)),
                ("a".to_string(), CustomCallAttribute::I64(1)),
            ]),
        );
    }

    #[test]
    fn test_custom_call_array_attribute() {
        // Arrays render with their element type, and empty arrays omit the element list.
        for (array, length, display, debug) in [
            (CustomCallArrayAttribute::I8(vec![-1, 2]), 2, "array<i8: -1, 2>", "I8([-1, 2])"),
            (CustomCallArrayAttribute::I16(vec![-1, 2]), 2, "array<i16: -1, 2>", "I16([-1, 2])"),
            (CustomCallArrayAttribute::I32(vec![-1, 2]), 2, "array<i32: -1, 2>", "I32([-1, 2])"),
            (CustomCallArrayAttribute::I64(vec![-1, 2]), 2, "array<i64: -1, 2>", "I64([-1, 2])"),
            (CustomCallArrayAttribute::U8(vec![1]), 1, "array<u8: 1>", "U8([1])"),
            (CustomCallArrayAttribute::U16(vec![1]), 1, "array<u16: 1>", "U16([1])"),
            (CustomCallArrayAttribute::U32(vec![1]), 1, "array<u32: 1>", "U32([1])"),
            (CustomCallArrayAttribute::U64(vec![1]), 1, "array<u64: 1>", "U64([1])"),
            (CustomCallArrayAttribute::F32(vec![1.5, -0.0]), 2, "array<f32: 1.5, -0.0>", "F32([1.5, -0.0])"),
            (
                CustomCallArrayAttribute::F64(vec![1.0, 2.5, 3.0]),
                3,
                "array<f64: 1.0, 2.5, 3.0>",
                "F64([1.0, 2.5, 3.0])",
            ),
            (CustomCallArrayAttribute::I32(Vec::new()), 0, "array<i32>", "I32([])"),
            (CustomCallArrayAttribute::F64(Vec::new()), 0, "array<f64>", "F64([])"),
        ] {
            assert_eq!(array.len(), length);
            assert_eq!(array.is_empty(), length == 0);
            assert_eq!(array.to_string(), display);
            assert_eq!(format!("{array:?}"), debug);
        }

        // Equal arrays hash identically, so they work as map keys.
        let array = CustomCallArrayAttribute::I32(vec![1, 2]);
        assert_eq!(array, array.clone());
        assert_eq!(hash_of(&array), hash_of(&CustomCallArrayAttribute::I32(vec![1, 2])));
        let arrays = HashMap::from([(array, "sizes"), (CustomCallArrayAttribute::F32(vec![0.5]), "scales")]);
        assert_eq!(arrays.get(&CustomCallArrayAttribute::I32(vec![1, 2])), Some(&"sizes"));
        assert_eq!(arrays.get(&CustomCallArrayAttribute::F32(vec![0.5])), Some(&"scales"));
        assert_eq!(arrays.get(&CustomCallArrayAttribute::I32(vec![2, 1])), None);

        // Floating-point elements compare bitwise, and arrays of different element types never compare equal.
        assert_eq!(CustomCallArrayAttribute::F32(vec![f32::NAN]), CustomCallArrayAttribute::F32(vec![f32::NAN]));
        assert_eq!(
            hash_of(&CustomCallArrayAttribute::F64(vec![f64::NAN])),
            hash_of(&CustomCallArrayAttribute::F64(vec![f64::NAN])),
        );
        assert_ne!(CustomCallArrayAttribute::F32(vec![-0.0]), CustomCallArrayAttribute::F32(vec![0.0]));
        assert_ne!(CustomCallArrayAttribute::F64(vec![-0.0]), CustomCallArrayAttribute::F64(vec![0.0]));
        assert_ne!(CustomCallArrayAttribute::F64(vec![1.0]), CustomCallArrayAttribute::F64(vec![1.0, 1.0]));
        assert_ne!(CustomCallArrayAttribute::I32(vec![1]), CustomCallArrayAttribute::I64(vec![1]));
        assert_ne!(CustomCallArrayAttribute::I8(Vec::new()), CustomCallArrayAttribute::U8(Vec::new()));
    }

    #[test]
    fn test_custom_call_input_output_alias() {
        let alias = CustomCallInputOutputAlias::new(0, 1);
        assert_eq!(alias.input_index(), 0);
        assert_eq!(alias.output_index(), 1);
        assert_eq!(alias.to_string(), "0->1");
        assert_eq!(format!("{alias:?}"), "CustomCallInputOutputAlias { input_index: 0, output_index: 1 }");
        assert_eq!(alias, CustomCallInputOutputAlias::new(0, 1));
        assert_ne!(alias, CustomCallInputOutputAlias::new(1, 0));
        assert_eq!(hash_of(&alias), hash_of(&CustomCallInputOutputAlias::new(0, 1)));
        let aliases = HashMap::from([(alias, "first"), (CustomCallInputOutputAlias::new(1, 0), "second")]);
        assert_eq!(aliases.get(&CustomCallInputOutputAlias::new(0, 1)), Some(&"first"));
        assert_eq!(aliases.get(&CustomCallInputOutputAlias::new(1, 0)), Some(&"second"));
        assert_eq!(aliases.get(&CustomCallInputOutputAlias::new(1, 1)), None);
    }

    #[test]
    fn test_custom_call_batching_display() {
        assert_eq!(CustomCallBatching::default(), CustomCallBatching::Rejected);
        for (batching, display, debug) in [
            (CustomCallBatching::Rejected, "rejected", "Rejected"),
            (CustomCallBatching::Sequential { unroll: None }, "sequential", "Sequential { unroll: None }"),
            (
                CustomCallBatching::Sequential { unroll: Some(2) },
                "sequential(unroll=2)",
                "Sequential { unroll: Some(2) }",
            ),
            (CustomCallBatching::SequentialUnrolled, "sequential_unrolled", "SequentialUnrolled"),
            (CustomCallBatching::BroadcastAll, "broadcast_all", "BroadcastAll"),
            (CustomCallBatching::ExpandDimensions, "expand_dimensions", "ExpandDimensions"),
            (CustomCallBatching::Vectorized, "vectorized", "Vectorized"),
        ] {
            assert_eq!(batching.to_string(), display);
            assert_eq!(format!("{batching:?}"), debug);
        }

        // Unroll factors distinguish sequential behaviors, and equal behaviors hash identically.
        assert_ne!(CustomCallBatching::Sequential { unroll: None }, CustomCallBatching::Sequential { unroll: Some(1) });
        assert_eq!(hash_of(&CustomCallBatching::BroadcastAll), hash_of(&CustomCallBatching::BroadcastAll));
        let behaviors = HashMap::from([
            (CustomCallBatching::Sequential { unroll: Some(2) }, "sequential"),
            (CustomCallBatching::Vectorized, "vectorized"),
        ]);
        assert_eq!(behaviors.get(&CustomCallBatching::Sequential { unroll: Some(2) }), Some(&"sequential"));
        assert_eq!(behaviors.get(&CustomCallBatching::Vectorized), Some(&"vectorized"));
        assert_eq!(behaviors.get(&CustomCallBatching::Sequential { unroll: None }), None);
    }

    #[test]
    fn test_custom_call_ragged_input_binding() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let binding = CustomCallRaggedInputBinding::new("data", 0, 1, 2, length.clone());
        assert_eq!(binding.name(), "data");
        assert_eq!(binding.input_index(), 0);
        assert_eq!(binding.axis(), 1);
        assert_eq!(binding.extent_input_index(), 2);
        assert_eq!(binding.dimension(), &length);
        assert_eq!(binding.to_string(), "data:input(0)@1<=input(2):length");
        assert_eq!(
            format!("{binding:?}"),
            "CustomCallRaggedInputBinding { name: \"data\", input_index: 0, axis: 1, extent_input_index: 2, \
             dimension: DimensionVariable { name: \"length\", bounds: DimensionBounds { lower: 0, upper: Some(5) }, \
             .. } }",
        );
        assert_eq!(binding, binding.clone());
        assert_ne!(binding, CustomCallRaggedInputBinding::new("data", 0, 0, 2, length.clone()));
        assert_eq!(hash_of(&binding), hash_of(&CustomCallRaggedInputBinding::new("data", 0, 1, 2, length.clone())));
        let bindings = HashMap::from([(binding, "data")]);
        assert_eq!(bindings.get(&CustomCallRaggedInputBinding::new("data", 0, 1, 2, length.clone())), Some(&"data"));
        assert_eq!(bindings.get(&CustomCallRaggedInputBinding::new("other", 0, 1, 2, length)), None);
    }

    #[test]
    fn test_custom_call_ragged_output_binding() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let preserved = CustomCallRaggedOutputBinding::Preserved { input_binding: "data".to_string(), axis: 0 };
        assert_eq!(preserved.to_string(), "preserve(data)@0");
        assert_eq!(format!("{preserved:?}"), "Preserved { input_binding: \"data\", axis: 0 }");
        let consumed = CustomCallRaggedOutputBinding::Consumed;
        assert_eq!(consumed.to_string(), "consume");
        assert_eq!(format!("{consumed:?}"), "Consumed");
        let fresh = CustomCallRaggedOutputBinding::Fresh { axis: 1, extent_output_index: 2, dimension: length.clone() };
        assert_eq!(fresh.to_string(), "fresh@1<=output(2):length");
        assert_eq!(
            format!("{fresh:?}"),
            "Fresh { axis: 1, extent_output_index: 2, dimension: DimensionVariable { name: \"length\", \
             bounds: DimensionBounds { lower: 0, upper: Some(5) }, .. } }",
        );

        assert_eq!(preserved, preserved.clone());
        assert_ne!(preserved, CustomCallRaggedOutputBinding::Preserved { input_binding: "data".to_string(), axis: 1 });
        assert_ne!(consumed, preserved);
        assert_eq!(hash_of(&fresh), hash_of(&fresh.clone()));
        let bindings = HashMap::from([(preserved.clone(), "preserved"), (consumed, "consumed"), (fresh, "fresh")]);
        assert_eq!(bindings.get(&preserved), Some(&"preserved"));
        assert_eq!(bindings.get(&CustomCallRaggedOutputBinding::Consumed), Some(&"consumed"));
        assert_eq!(
            bindings.get(&CustomCallRaggedOutputBinding::Fresh { axis: 1, extent_output_index: 2, dimension: length }),
            Some(&"fresh"),
        );
    }

    #[test]
    fn test_custom_call_ragged_contract() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let contract = preserved_ragged_contract(length.clone());
        assert_eq!(contract.input_bindings(), &[CustomCallRaggedInputBinding::new("data", 0, 0, 1, length.clone())]);
        assert_eq!(
            contract.output_bindings(),
            &[CustomCallRaggedOutputBinding::Preserved { input_binding: "data".to_string(), axis: 0 }],
        );
        assert_eq!(contract.to_string(), "{inputs=[data:input(0)@0<=input(1):length], outputs=[preserve(data)@0]}");
        assert_eq!(
            format!("{contract:?}"),
            "CustomCallRaggedContract { input_bindings: [CustomCallRaggedInputBinding { name: \"data\", \
             input_index: 0, axis: 0, extent_input_index: 1, dimension: DimensionVariable { name: \"length\", \
             bounds: DimensionBounds { lower: 0, upper: Some(5) }, .. } }], output_bindings: [Preserved { \
             input_binding: \"data\", axis: 0 }], batch_prefix_count: 0, ragged_discharged: false }",
        );

        // Several bindings render in declaration order. Batch-prefixing shifts every bound axis and records the
        // accumulated prefix, and both batching transitions render their recorded state.
        let count = DimensionVariable::new("count", DimensionBounds::new(0, Some(3)).unwrap());
        let contract = CustomCallRaggedContract::new(
            vec![
                CustomCallRaggedInputBinding::new("lhs", 0, 0, 2, length.clone()),
                CustomCallRaggedInputBinding::new("rhs", 1, 1, 2, length.clone()),
            ],
            vec![
                CustomCallRaggedOutputBinding::Preserved { input_binding: "lhs".to_string(), axis: 0 },
                CustomCallRaggedOutputBinding::Consumed,
                CustomCallRaggedOutputBinding::Fresh { axis: 1, extent_output_index: 3, dimension: count },
            ],
        );
        assert_eq!(
            contract.to_string(),
            "{inputs=[lhs:input(0)@0<=input(2):length, rhs:input(1)@1<=input(2):length], \
             outputs=[preserve(lhs)@0, consume, fresh@1<=output(3):count]}",
        );
        assert_eq!(
            contract.batch_prefixed(false).to_string(),
            "{inputs=[lhs:input(0)@1<=input(2):length, rhs:input(1)@2<=input(2):length], \
             outputs=[preserve(lhs)@1, consume, fresh@2<=output(3):count], batch_prefix_count=1}",
        );
        assert_eq!(
            contract.batch_prefixed(true).batch_prefixed(false).to_string(),
            "{inputs=[lhs:input(0)@2<=input(2):length, rhs:input(1)@3<=input(2):length], \
             outputs=[preserve(lhs)@2, consume, fresh@3<=output(3):count], batch_prefix_count=2, \
             ragged_discharged=true}",
        );
        assert_eq!(
            contract.ragged_discharged().to_string(),
            "{inputs=[lhs:input(0)@0<=input(2):length, rhs:input(1)@1<=input(2):length], \
             outputs=[preserve(lhs)@0, consume, fresh@1<=output(3):count], ragged_discharged=true}",
        );
        assert_eq!(CustomCallRaggedContract::new(Vec::new(), Vec::new()).to_string(), "{inputs=[], outputs=[]}");

        // The recorded batching state participates in equality and hashing.
        assert_eq!(contract, contract.clone());
        assert_ne!(contract, contract.ragged_discharged());
        assert_ne!(contract, contract.batch_prefixed(false));
        assert_eq!(hash_of(&contract), hash_of(&contract.clone()));
        let contracts = HashMap::from([(contract.clone(), "unbatched"), (contract.batch_prefixed(false), "batched")]);
        assert_eq!(contracts.get(&contract), Some(&"unbatched"));
        assert_eq!(contracts.get(&contract.batch_prefixed(false)), Some(&"batched"));
        assert_eq!(contracts.get(&contract.ragged_discharged()), None);
    }

    #[test]
    fn test_custom_call_ragged_contract_consumed_dimensions() {
        // A dimension shared by several active bindings is consumed only when no binding that carries it is
        // preserved, and it is reported once.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let contract = CustomCallRaggedContract::new(
            vec![
                CustomCallRaggedInputBinding::new("lhs", 0, 0, 2, length.clone()),
                CustomCallRaggedInputBinding::new("rhs", 1, 0, 2, length.clone()),
            ],
            vec![CustomCallRaggedOutputBinding::Preserved { input_binding: "rhs".to_string(), axis: 0 }],
        );
        let active = vec![
            ("lhs".to_string(), RaggedAxis::new(0, Array::scalar(2i32).unwrap(), length.clone(), Vec::new())),
            ("rhs".to_string(), RaggedAxis::new(0, Array::scalar(2i32).unwrap(), length.clone(), Vec::new())),
        ];
        assert_eq!(contract.consumed_dimensions(active.as_slice()), Vec::new());

        let consumed_contract = CustomCallRaggedContract::new(
            contract.input_bindings().to_vec(),
            vec![CustomCallRaggedOutputBinding::Consumed],
        );
        assert_eq!(consumed_contract.consumed_dimensions(active.as_slice()), vec![length]);
        assert_eq!(consumed_contract.consumed_dimensions::<Array>(&[]), Vec::new());
    }

    #[test]
    fn test_custom_call() {
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])]);
        assert_eq!(operation.name(), CUSTOM_CALL_OPERATION_NAME);
        assert_eq!(operation.target_name(), "ryft.test.add_one");
        assert_eq!(operation.output_types(), &[ArrayType::new_static(DataType::F32, [2])]);
        assert!(operation.attributes().is_empty());
        assert!(operation.input_layouts().is_empty());
        assert!(operation.input_output_aliases().is_empty());
        assert!(!operation.has_side_effect());
        assert_eq!(operation.effect_class(), None);
        assert!(operation.effects().is_pure());
        assert_eq!(operation.batching(), CustomCallBatching::Rejected);
        assert_eq!(operation.ragged_contract(), None);
        assert_eq!(operation.to_string(), "custom_call [target=ryft.test.add_one]");

        // Operations compare and hash by their complete payload.
        assert_eq!(operation, operation.clone());
        assert_eq!(hash_of(&operation), hash_of(&operation.clone()));
        assert_ne!(operation, CustomCallOperation::new("ryft.test.add_two", operation.output_types().to_vec()));
        assert_ne!(operation, operation.clone().with_batching(CustomCallBatching::BroadcastAll));
        assert_eq!(
            format!("{:?}", CustomCallOperation::new("kernel", Vec::new())),
            "CustomCallOperation { target_name: \"kernel\", output_types: [], attributes: [], input_layouts: [], \
             input_output_aliases: [], effect_class: None, batching: Rejected, ragged_contract: None }",
        );

        // Every configured field renders in declaration order, one field per line once the section is long.
        let operation = operation
            .with_attribute("scale", 2.0)
            .with_input_layouts([None])
            .with_input_output_alias(0, 0)
            .unwrap()
            .with_effect_class(EffectClass::DeviceOrderedIo)
            .with_batching(CustomCallBatching::BroadcastAll)
            .with_ragged_contract(CustomCallRaggedContract::new(
                Vec::new(),
                vec![CustomCallRaggedOutputBinding::Consumed],
            ));
        assert_eq!(
            operation.to_string(),
            indoc! {"
                custom_call [
                    target=ryft.test.add_one,
                    scale=2.0,
                    input_layouts=[default],
                    input_output_alias=0->0,
                    has_side_effect=true,
                    effect_class=device_ordered_io,
                    batching=broadcast_all,
                    ragged_contract={inputs=[], outputs=[consume]},
                ]
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_with_attribute() {
        let operation = CustomCallOperation::new("kernel", Vec::new())
            .with_attribute("scale", 2.5f32)
            .with_attribute("count", 4i64)
            .with_attribute("label", "x");
        assert_eq!(
            operation.attributes(),
            &[
                ("scale".to_string(), CustomCallAttribute::F32(2.5)),
                ("count".to_string(), CustomCallAttribute::I64(4)),
                ("label".to_string(), CustomCallAttribute::String("x".to_string())),
            ],
        );
        assert_eq!(operation.to_string(), "custom_call [target=kernel, scale=2.5f32, count=4, label=x]");

        // Nested dictionaries and arrays render inline, and long attribute lists wrap onto one line per field.
        let operation = operation
            .with_attribute("sizes", vec![1i32, 2])
            .with_attribute(
                "options",
                CustomCallAttribute::Dictionary(vec![
                    ("verbose".to_string(), true.into()),
                    ("offset".to_string(), 7u8.into()),
                ]),
            )
            .with_attribute("payload", vec![0u8, 255]);
        assert_eq!(
            operation.to_string(),
            indoc! {"
                custom_call [
                    target=kernel,
                    scale=2.5f32,
                    count=4,
                    label=x,
                    sizes=array<i32: 1, 2>,
                    options={verbose=true, offset=7u8},
                    payload=bytes [00, ff],
                ]
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_with_input_layouts() {
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()));
        let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
            .with_input_layouts([Some(column_major.clone()), None]);
        assert_eq!(operation.input_layouts(), &[Some(column_major.clone()), None]);
        assert_eq!(operation.to_string(), "custom_call [target=kernel, input_layouts=[tiled{0,1}, default]]");

        // A later declaration replaces the earlier one, and an empty declaration removes the field.
        let strided = Layout::Strided(StridedLayout::new(vec![4]));
        let operation = operation.with_input_layouts([Some(strided.clone())]);
        assert_eq!(operation.input_layouts(), &[Some(strided)]);
        assert_eq!(operation.to_string(), "custom_call [target=kernel, input_layouts=[strided{4}]]");
        let operation = operation.with_input_layouts(Vec::new());
        assert!(operation.input_layouts().is_empty());
        assert_eq!(operation.to_string(), "custom_call [target=kernel]");
    }

    #[test]
    fn test_custom_call_with_input_output_alias() {
        let operation = CustomCallOperation::new(
            "kernel",
            vec![ArrayType::new_static(DataType::F32, [2]), ArrayType::new_static(DataType::F32, [3])],
        )
        .with_input_output_alias(0, 0)
        .unwrap()
        .with_input_output_alias(2, 1)
        .unwrap();
        assert_eq!(
            operation.input_output_aliases(),
            &[CustomCallInputOutputAlias::new(0, 0), CustomCallInputOutputAlias::new(2, 1)],
        );
        assert_eq!(
            operation.to_string(),
            "custom_call [target=kernel, input_output_alias=0->0, input_output_alias=2->1]",
        );

        // Each input and each output participates in at most one alias.
        assert!(matches!(
            operation.clone().with_input_output_alias(0, 2),
            Err(TypeError::Invalid { message })
                if message == "`custom_call` cannot add alias 0->2 because alias `0->0` already uses the same input \
                               or output",
        ));
        assert!(matches!(
            operation.with_input_output_alias(1, 1),
            Err(TypeError::Invalid { message })
                if message == "`custom_call` cannot add alias 1->1 because alias `2->1` already uses the same input \
                               or output",
        ));
    }

    #[test]
    fn test_custom_call_with_side_effect() {
        let operation = CustomCallOperation::new("kernel", Vec::new()).with_side_effect();
        assert!(operation.has_side_effect());
        assert_eq!(operation.effect_class(), Some(EffectClass::OrderedIo));
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedIo));
        // The global ordered class renders as the bare `has_side_effect` flag.
        assert_eq!(operation.to_string(), "custom_call [target=kernel, has_side_effect=true]");
        // The side effect replaces any earlier effect selection.
        assert_eq!(
            operation.with_effect_class(EffectClass::UnorderedIo).with_side_effect().effect_class(),
            Some(EffectClass::OrderedIo),
        );
    }

    #[test]
    fn test_custom_call_with_effect_class() {
        // Classes other than the global ordered class render explicitly beside the side-effect flag.
        let device_ordered =
            CustomCallOperation::new("kernel", Vec::new()).with_effect_class(EffectClass::DeviceOrderedIo);
        assert!(device_ordered.has_side_effect());
        assert_eq!(device_ordered.effect_class(), Some(EffectClass::DeviceOrderedIo));
        assert_eq!(device_ordered.effects().classes(), EffectClasses::single(EffectClass::DeviceOrderedIo));
        assert_eq!(
            device_ordered.to_string(),
            "custom_call [target=kernel, has_side_effect=true, effect_class=device_ordered_io]",
        );

        let unordered = CustomCallOperation::new("kernel", Vec::new()).with_effect_class(EffectClass::UnorderedIo);
        assert!(unordered.has_side_effect());
        assert_eq!(unordered.effect_class(), Some(EffectClass::UnorderedIo));
        assert_eq!(unordered.effects().classes(), EffectClasses::single(EffectClass::UnorderedIo));
        assert_eq!(
            unordered.to_string(),
            "custom_call [target=kernel, has_side_effect=true, effect_class=unordered_io]"
        );

        // The last selection wins.
        assert_eq!(unordered.with_effect_class(EffectClass::OrderedIo).effect_class(), Some(EffectClass::OrderedIo));
    }

    #[test]
    fn test_custom_call_with_batching() {
        // The batching behavior renders only when it differs from the default `Rejected` behavior.
        let operation = CustomCallOperation::new("kernel", Vec::new());
        assert_eq!(operation.batching(), CustomCallBatching::Rejected);
        assert_eq!(operation.to_string(), "custom_call [target=kernel]");
        let operation = operation.with_batching(CustomCallBatching::Sequential { unroll: None });
        assert_eq!(operation.batching(), CustomCallBatching::Sequential { unroll: None });
        assert_eq!(operation.to_string(), "custom_call [target=kernel, batching=sequential]");
        let operation = operation.with_batching(CustomCallBatching::Sequential { unroll: Some(2) });
        assert_eq!(operation.to_string(), "custom_call [target=kernel, batching=sequential(unroll=2)]");
        let operation = operation.with_batching(CustomCallBatching::SequentialUnrolled);
        assert_eq!(operation.to_string(), "custom_call [target=kernel, batching=sequential_unrolled]");
        let operation = operation.with_batching(CustomCallBatching::BroadcastAll).with_side_effect();
        assert_eq!(operation.batching(), CustomCallBatching::BroadcastAll);
        assert_eq!(operation.to_string(), "custom_call [target=kernel, has_side_effect=true, batching=broadcast_all]");
        let operation = operation.with_batching(CustomCallBatching::Rejected);
        assert_eq!(operation.to_string(), "custom_call [target=kernel, has_side_effect=true]");
    }

    #[test]
    fn test_custom_call_with_ragged_contract() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let contract = preserved_ragged_contract(length);
        let operation = CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [4])])
            .with_batching(CustomCallBatching::BroadcastAll)
            .with_ragged_contract(contract.clone());
        assert_eq!(operation.ragged_contract(), Some(&contract));
        assert_eq!(
            operation.to_string(),
            indoc! {"
                custom_call [
                    target=ryft.test.ragged,
                    batching=broadcast_all,
                    ragged_contract={inputs=[data:input(0)@0<=input(1):length], \
                outputs=[preserve(data)@0]},
                ]
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_type_inference() {
        // Declared output types do not depend on the inputs, which the kernel receives verbatim.
        check_operation_type_inference!(
            operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])]),
            cases = [
                { input_types = [], output_types = [ArrayType::new_static(DataType::F32, [2])] },
                {
                    input_types = [ArrayType::scalar(DataType::I32), ArrayType::new_static(DataType::F64, [3])],
                    output_types = [ArrayType::new_static(DataType::F32, [2])],
                },
            ],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("kernel", Vec::new()),
            cases = [{ input_types = [], output_types = [] }],
        );

        // Aliases must refer to an existing input and output of identical types.
        check_operation_type_inference!(
            operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
                .with_input_output_alias(0, 0)
                .unwrap(),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2])],
                    output_types = [ArrayType::new_static(DataType::F32, [2])],
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    error = "`custom_call` alias `0->0` requires matching input and output types but input 0 has \
                             type `f32[]` and output 0 has type `f32[2]`",
                },
                {
                    input_types = [],
                    error = "`custom_call` alias `0->0` refers to input 0 but the call has 0 array inputs",
                },
            ],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
                .with_input_output_alias(0, 1)
                .unwrap(),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [2])],
                error = "`custom_call` alias `0->1` refers to output 1 but the call has 1 outputs",
            }],
        );

        // Custom calls are region-free.
        assert_eq!(
            CustomCallOperation::new("kernel", Vec::new())
                .infer_output_types(&[], &[RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE)]),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_custom_call_type_inference_dynamic_outputs() {
        // The homogeneous form has no way to ground a dynamic result extent, because only the mixed form accepts the
        // trailing first-class dimension inputs that define one.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        check_operation_type_inference!(
            operation = CustomCallOperation::new(
                "ryft.test.dynamic",
                vec![
                    ArrayType::new_static(DataType::F32, [2]),
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(3)])),
                ],
            ),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [2])],
                error = "`custom_call` requires explicit result-extent inputs for dynamic output type `f32[rows, 3]`",
            }],
        );
    }

    #[test]
    fn test_custom_call_type_inference_array_ir() {
        // Composite output extents are positional instruction inputs: output-major and then axis-major.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(2, Some(17)).unwrap());
        let dynamic_output_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(rows.clone()),
                Dimension::Static(3),
                Dimension::Dynamic(columns.clone()),
            ]),
        );
        let operation = CustomCallOperation::new("ryft.test.dynamic", vec![dynamic_output_type.clone()]);
        assert_eq!(
            operation.infer_parent_output_types(
                &[
                    ArrayType::new_static(DataType::F32, [2]).into(),
                    DimensionType::from(rows.clone()).into(),
                    DimensionType::from(columns.clone()).into(),
                ],
                &[],
            ),
            Ok(vec![dynamic_output_type.clone().into()]),
        );
        assert_eq!(
            operation.infer_parent_output_types(
                &[
                    ArrayType::new_static(DataType::F32, [2]).into(),
                    DimensionType::from(columns.clone()).into(),
                    DimensionType::from(rows.clone()).into(),
                ],
                &[],
            ),
            Err(TypeError::invalid(
                "`custom_call` output-extent input defines dimension variable `columns`, but the corresponding \
                 declared output axis refers to `rows`",
            )),
        );
        assert_eq!(
            operation.infer_parent_output_types(
                &[DimensionType::new("extent", DimensionBounds::new(1, Some(9)).unwrap()).into()],
                &[],
            ),
            Err(TypeError::invalid(
                "`custom_call` expects 2 trailing output-extent dimensions but only 1 inputs were provided",
            )),
        );

        // Aliases count only the leading array inputs, so a dynamic aliased output validates against its array input.
        assert_eq!(
            operation.clone().with_input_output_alias(0, 0).unwrap().infer_parent_output_types(
                &[
                    dynamic_output_type.clone().into(),
                    DimensionType::from(rows.clone()).into(),
                    DimensionType::from(columns.clone()).into(),
                ],
                &[],
            ),
            Ok(vec![dynamic_output_type.into()]),
        );

        // Custom calls are region-free in the mixed universe too.
        let region = RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE);
        assert_eq!(operation.infer_parent_region_input_types(&[], &[]), Ok(Vec::new()));
        assert_eq!(
            operation.infer_parent_region_input_types(&[], std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
        assert_eq!(
            operation.infer_parent_output_types(
                &[DimensionType::from(rows).into(), DimensionType::from(columns).into()],
                std::slice::from_ref(&region),
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_custom_call_type_inference_manual_variation() {
        // Opaque code cannot be assumed to agree across devices, so an output declared without a sharding inherits
        // the inputs' manual variation, placed replicated on their mesh, whereas a declared sharding is kept as is.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap();
        let varying_input = ArrayType::new_static(DataType::F32, [2]).with_sharding(varying.clone()).unwrap();
        let invariant_input =
            ArrayType::new_static(DataType::F32, [2]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let operation = CustomCallOperation::new(
            "ryft.test.add_one",
            vec![ArrayType::new_static(DataType::F32, [2]), invariant_input.clone()],
        );
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [varying_input.clone()],
                    output_types = [varying_input.clone(), invariant_input.clone()],
                },
                // Invariant and unsharded inputs leave undeclared outputs unsharded.
                {
                    input_types = [invariant_input.clone()],
                    output_types = [ArrayType::new_static(DataType::F32, [2]), invariant_input.clone()],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2])],
                    output_types = [ArrayType::new_static(DataType::F32, [2]), invariant_input.clone()],
                },
            ],
        );

        // The mixed universe applies the same rule, including to dynamic outputs.
        assert_eq!(
            operation.infer_parent_output_types(&[varying_input.clone().into()], &[]),
            Ok(vec![varying_input.clone().into(), invariant_input.into()]),
        );
        let length = DimensionType::new("length", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![length.to_dimension()]));
        let operation = CustomCallOperation::new("ryft.test.dynamic", vec![dynamic_type.clone()]);
        assert_eq!(
            operation.infer_parent_output_types(&[varying_input.into(), length.into()], &[]),
            Ok(vec![dynamic_type.with_sharding(varying).unwrap().into()]),
        );
    }

    #[test]
    fn test_custom_call_type_inference_attribute_names() {
        // Attribute names must be unique within each dictionary, including nested ones, in both universes.
        check_operation_type_inference!(
            operation = CustomCallOperation::new("kernel", Vec::new())
                .with_attribute("scale", 1.0)
                .with_attribute("scale", 2.0),
            cases = [{ input_types = [], error = "`custom_call` declares attribute `scale` more than once" }],
        );
        let nested = CustomCallOperation::new("kernel", Vec::new()).with_attribute(
            "options",
            CustomCallAttribute::Dictionary(vec![
                ("mode".to_string(), "fast".into()),
                ("mode".to_string(), "slow".into()),
            ]),
        );
        check_operation_type_inference!(
            operation = nested.clone(),
            cases = [{ input_types = [], error = "`custom_call` declares attribute `mode` more than once" }],
        );
        assert_eq!(
            nested.infer_parent_output_types(&[], &[]),
            Err(TypeError::invalid("`custom_call` declares attribute `mode` more than once")),
        );

        // The same name may appear at different nesting levels.
        check_operation_type_inference!(
            operation = CustomCallOperation::new("kernel", Vec::new()).with_attribute("mode", "fast").with_attribute(
                "options",
                CustomCallAttribute::Dictionary(vec![("mode".to_string(), "slow".into())]),
            ),
            cases = [{ input_types = [], output_types = [] }],
        );
    }

    #[test]
    fn test_custom_call_type_inference_input_layouts() {
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()));
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3]);
        check_operation_type_inference!(
            operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
                .with_input_layouts([Some(column_major.clone()), None]),
            cases = [
                {
                    input_types = [matrix_type.clone(), ArrayType::scalar(DataType::I32)],
                    output_types = [ArrayType::new_static(DataType::F32, [2])],
                },
                {
                    input_types = [matrix_type.clone()],
                    error = "`custom_call` declares 2 input layouts but the call has 1 array inputs",
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2]), ArrayType::scalar(DataType::I32)],
                    error = "`custom_call` input layout `tiled{0,1}` has rank 2 but input 0 has type `f32[2]`",
                },
            ],
        );

        // A declared layout of an aliased input must match the layout of its output.
        check_operation_type_inference!(
            operation = CustomCallOperation::new("kernel", vec![matrix_type.clone()])
                .with_input_layouts([Some(column_major.clone())])
                .with_input_output_alias(0, 0)
                .unwrap(),
            cases = [{
                input_types = [matrix_type.clone()],
                error = "`custom_call` alias `0->0` requires the declared layout `tiled{0,1}` of input 0 to match the \
                         layout of output 0",
            }],
        );
        let column_major_type = matrix_type.clone().with_layout(column_major.clone());
        check_operation_type_inference!(
            operation = CustomCallOperation::new("kernel", vec![column_major_type.clone()])
                .with_input_layouts([Some(column_major.clone())])
                .with_input_output_alias(0, 0)
                .unwrap(),
            cases = [{ input_types = [column_major_type.clone()], output_types = [column_major_type] }],
        );

        // Mixed calls declare layouts only for their leading array inputs, excluding trailing dimension inputs.
        let length = DimensionType::new("length", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![length.to_dimension()]));
        let operation = CustomCallOperation::new("ryft.test.dynamic", vec![dynamic_type.clone()])
            .with_input_layouts([Some(column_major)]);
        assert_eq!(
            operation.infer_parent_output_types(&[matrix_type.clone().into(), length.clone().into()], &[]),
            Ok(vec![dynamic_type.into()]),
        );
        assert_eq!(
            operation.infer_parent_output_types(&[matrix_type.clone().into(), matrix_type.into(), length.into()], &[]),
            Err(TypeError::invalid("`custom_call` declares 1 input layouts but the call has 2 array inputs")),
        );
    }

    #[test]
    fn test_custom_call_type_inference_ragged_contract() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let packed_type = ArrayType::new_static(DataType::F32, [4]);
        let extent_type = ArrayType::scalar(DataType::I32);
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_ragged_contract(preserved_ragged_contract(length.clone())),
            cases = [
                { input_types = [packed_type.clone(), extent_type.clone()], output_types = [packed_type.clone()] },
                {
                    input_types = [packed_type.clone(), ArrayType::scalar(DataType::F32)],
                    error = "`custom_call` ragged input binding `data` requires extent input 1 to be an integer \
                             scalar but got `f32[]`",
                },
                {
                    input_types = [packed_type.clone()],
                    error = "`custom_call` ragged input binding `data` refers to extent input 1 but the call has 1 \
                             array inputs",
                },
                {
                    input_types = [],
                    error = "`custom_call` ragged input binding `data` refers to input 0 but the call has 0 array \
                             inputs",
                },
            ],
        );

        // The contract must declare one output binding per output.
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()]).with_ragged_contract(
                CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new("data", 0, 0, 1, length.clone())],
                    Vec::new(),
                ),
            ),
            cases = [{
                input_types = [packed_type.clone(), extent_type.clone()],
                error = "`custom_call` ragged contract declares 0 output bindings but the call has 1 outputs",
            }],
        );

        // Input bindings must have unique names and inputs, and preserved outputs must name an existing binding.
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()]).with_ragged_contract(
                CustomCallRaggedContract::new(
                    vec![
                        CustomCallRaggedInputBinding::new("data", 0, 0, 1, length.clone()),
                        CustomCallRaggedInputBinding::new("data", 2, 0, 1, length.clone()),
                    ],
                    vec![CustomCallRaggedOutputBinding::Consumed],
                ),
            ),
            cases = [{
                input_types = [packed_type.clone(), extent_type.clone(), packed_type.clone()],
                error = "`custom_call` ragged input binding `data` duplicates binding `data`",
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()]).with_ragged_contract(
                CustomCallRaggedContract::new(
                    vec![
                        CustomCallRaggedInputBinding::new("left", 0, 0, 1, length.clone()),
                        CustomCallRaggedInputBinding::new("right", 0, 0, 1, length.clone()),
                    ],
                    vec![CustomCallRaggedOutputBinding::Preserved { input_binding: "left".to_string(), axis: 0 }],
                ),
            ),
            cases = [{
                input_types = [packed_type.clone(), extent_type.clone()],
                error = "`custom_call` ragged input bindings `left` and `right` both bind input 0",
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()]).with_ragged_contract(
                CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new("data", 0, 0, 1, length.clone())],
                    vec![CustomCallRaggedOutputBinding::Preserved { input_binding: "missing".to_string(), axis: 0 }],
                ),
            ),
            cases = [{
                input_types = [packed_type.clone(), extent_type.clone()],
                error = "`custom_call` ragged output 0 preserves unknown input binding `missing`",
            }],
        );

        // Bound axes must exist and have a finite static extent that covers the dimension's upper bound.
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()]).with_ragged_contract(
                preserved_ragged_contract(DimensionVariable::new("length", DimensionBounds::new(0, Some(6)).unwrap())),
            ),
            cases = [{
                input_types = [packed_type.clone(), extent_type.clone()],
                error = "`custom_call` ragged dimension `length` with bounds [0, 6) exceeds the physical extent 4 of \
                         input 0 axis 0",
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()]).with_ragged_contract(
                CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new("data", 0, 1, 1, length.clone())],
                    vec![CustomCallRaggedOutputBinding::Consumed],
                ),
            ),
            cases = [{
                input_types = [packed_type.clone(), extent_type.clone()],
                error = "`custom_call` ragged contract input 0 axis 1 is out of bounds for type `f32[4]`",
            }],
        );
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_ragged_contract(preserved_ragged_contract(length.clone())),
            cases = [{
                input_types = [
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)])),
                    extent_type.clone(),
                ],
                error = "`custom_call` ragged contract input 0 axis 0 must have a finite static physical bound but \
                         has dimension `rows`",
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()]).with_ragged_contract(
                CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new(
                        "data",
                        0,
                        0,
                        1,
                        DimensionVariable::new("length", DimensionBounds::unbounded()),
                    )],
                    vec![CustomCallRaggedOutputBinding::Consumed],
                ),
            ),
            cases = [{
                input_types = [packed_type.clone(), extent_type.clone()],
                error = "`custom_call` ragged dimension `length` with bounds [0, ∞) exceeds the physical extent 4 of \
                         input 0 axis 0",
            }],
        );

        // Alias preservation requires the same bound input and physical axis. Consumed and fresh outputs cannot alias
        // a ragged-bound input, but an alias to an unrelated dense input remains valid.
        let matrix_type = ArrayType::new_static(DataType::F32, [4, 4]);
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![matrix_type.clone()])
                .with_input_output_alias(0, 0)
                .unwrap()
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new("data", 0, 0, 1, length.clone())],
                    vec![CustomCallRaggedOutputBinding::Preserved { input_binding: "data".to_string(), axis: 1 }],
                )),
            cases = [{
                input_types = [matrix_type.clone(), extent_type.clone()],
                error = "`custom_call` alias `0->0` conflicts with preserved ragged binding `data` because aliases \
                         require the same packed input, physical axis, dimension identity, and extent binding",
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_input_output_alias(0, 0)
                .unwrap()
                .with_ragged_contract(preserved_ragged_contract(length.clone())),
            cases = [{
                input_types = [packed_type.clone(), extent_type.clone()],
                output_types = [packed_type.clone()],
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_input_output_alias(0, 0)
                .unwrap()
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new("data", 0, 0, 1, length.clone())],
                    vec![CustomCallRaggedOutputBinding::Consumed],
                )),
            cases = [{
                input_types = [packed_type.clone(), extent_type.clone()],
                error = "`custom_call` consumed ragged output 0 cannot retain alias `0->0`",
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_input_output_alias(1, 0)
                .unwrap()
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new("data", 0, 0, 2, length.clone())],
                    vec![CustomCallRaggedOutputBinding::Consumed],
                )),
            cases = [{
                input_types = [packed_type.clone(), packed_type.clone(), extent_type.clone()],
                output_types = [packed_type.clone()],
            }],
        );

        // Fresh outputs need a fresh identity and an existing integer extent output, and cannot alias bound inputs.
        let output_length = DimensionVariable::new("output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let fresh_output_types = vec![packed_type.clone(), extent_type.clone()];
        let fresh_contract = CustomCallRaggedContract::new(
            vec![CustomCallRaggedInputBinding::new("data", 0, 0, 2, length.clone())],
            vec![
                CustomCallRaggedOutputBinding::Fresh {
                    axis: 0,
                    extent_output_index: 1,
                    dimension: output_length.clone(),
                },
                CustomCallRaggedOutputBinding::Consumed,
            ],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.fresh_ragged", fresh_output_types.clone())
                .with_input_output_alias(1, 0)
                .unwrap()
                .with_ragged_contract(fresh_contract.clone()),
            cases = [{
                input_types = [packed_type.clone(), packed_type.clone(), extent_type.clone()],
                output_types = [packed_type.clone(), extent_type.clone()],
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.fresh_ragged", fresh_output_types.clone())
                .with_input_output_alias(0, 0)
                .unwrap()
                .with_ragged_contract(fresh_contract),
            cases = [{
                input_types = [packed_type.clone(), packed_type.clone(), extent_type.clone()],
                error = "`custom_call` fresh ragged output 0 cannot retain alias `0->0`",
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new(
                "ryft.test.fresh_ragged",
                vec![packed_type.clone(), ArrayType::scalar(DataType::F32)],
            )
            .with_ragged_contract(CustomCallRaggedContract::new(
                Vec::new(),
                vec![
                    CustomCallRaggedOutputBinding::Fresh {
                        axis: 0,
                        extent_output_index: 1,
                        dimension: output_length.clone(),
                    },
                    CustomCallRaggedOutputBinding::Consumed,
                ],
            )),
            cases = [{
                input_types = [],
                error = "`custom_call` fresh ragged output 0 requires extent output 1 to be an integer scalar but got \
                         `f32[]`",
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.fresh_ragged", vec![packed_type.clone()])
                .with_ragged_contract(CustomCallRaggedContract::new(
                    Vec::new(),
                    vec![CustomCallRaggedOutputBinding::Fresh {
                        axis: 0,
                        extent_output_index: 1,
                        dimension: output_length,
                    }],
                )),
            cases = [{
                input_types = [],
                error = "`custom_call` fresh ragged output 0 refers to extent output 1 but the call has 1 outputs",
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.fresh_ragged", fresh_output_types)
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new("data", 0, 0, 1, length.clone())],
                    vec![
                        CustomCallRaggedOutputBinding::Fresh { axis: 0, extent_output_index: 1, dimension: length },
                        CustomCallRaggedOutputBinding::Consumed,
                    ],
                )),
            cases = [{
                input_types = [packed_type, extent_type],
                error = "`custom_call` fresh ragged output 0 dimension `length` is already declared by input binding \
                         `data`",
            }],
        );
    }

    #[test]
    fn test_custom_call_type_inference_ragged_contract_batch_prefix() {
        // After repeated dense batching, extent inputs and outputs must carry the complete accumulated batch prefix.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3, 4]);
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()]).with_ragged_contract(
                preserved_ragged_contract(length.clone()).batch_prefixed(false).batch_prefixed(false),
            ),
            cases = [
                {
                    input_types = [packed_type.clone(), ArrayType::new_static(DataType::I32, [2, 3])],
                    output_types = [packed_type.clone()],
                },
                {
                    input_types = [packed_type.clone(), ArrayType::new_static(DataType::I32, [2])],
                    error = "`custom_call` ragged input binding `data` requires extent input 1 to be a rank-2 \
                             batch-prefixed integer tensor but got `i32[2]`",
                },
            ],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [2, 4])])
                .with_ragged_contract(preserved_ragged_contract(length.clone()).batch_prefixed(false)),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [2, 4]), ArrayType::scalar(DataType::I32)],
                error = "`custom_call` ragged input binding `data` requires extent input 1 to be a batch-prefixed \
                         integer vector but got `i32[]`",
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new(
                "ryft.test.fresh_ragged",
                vec![packed_type.clone(), ArrayType::new_static(DataType::I32, [2])],
            )
            .with_ragged_contract(
                CustomCallRaggedContract::new(
                    Vec::new(),
                    vec![
                        CustomCallRaggedOutputBinding::Fresh { axis: 0, extent_output_index: 1, dimension: length },
                        CustomCallRaggedOutputBinding::Consumed,
                    ],
                )
                .batch_prefixed(false)
                .batch_prefixed(false),
            ),
            cases = [{
                input_types = [packed_type],
                error = "`custom_call` fresh ragged output 0 requires extent output 1 to be a rank-2 batch-prefixed \
                         integer tensor but got `i32[2]`",
            }],
        );
    }

    #[test]
    fn test_custom_call_type_inference_ragged_contract_shared_dimensions() {
        // Bindings may share a dimension identity only when they also share its extent source.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let packed_type = ArrayType::new_static(DataType::F32, [4]);
        let extent_type = ArrayType::scalar(DataType::I32);
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.shared_input_dimension", vec![packed_type.clone()])
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![
                        CustomCallRaggedInputBinding::new("lhs", 0, 0, 2, length.clone()),
                        CustomCallRaggedInputBinding::new("rhs", 1, 0, 2, length.clone()),
                    ],
                    vec![CustomCallRaggedOutputBinding::Consumed],
                )),
            cases = [{
                input_types = [packed_type.clone(), packed_type.clone(), extent_type.clone(), extent_type.clone()],
                output_types = [packed_type.clone()],
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.shared_input_dimension", vec![packed_type.clone()])
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![
                        CustomCallRaggedInputBinding::new("lhs", 0, 0, 2, length.clone()),
                        CustomCallRaggedInputBinding::new("rhs", 1, 0, 3, length.clone()),
                    ],
                    vec![CustomCallRaggedOutputBinding::Consumed],
                )),
            cases = [{
                input_types = [packed_type.clone(), packed_type.clone(), extent_type.clone(), extent_type.clone()],
                error = "`custom_call` ragged input bindings `lhs` and `rhs` reuse dimension `length` with different \
                         extent inputs 2 and 3",
            }],
        );

        let output_types = vec![packed_type.clone(), extent_type.clone(), packed_type, extent_type];
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.shared_output_dimension", output_types.clone())
                .with_ragged_contract(CustomCallRaggedContract::new(
                    Vec::new(),
                    vec![
                        CustomCallRaggedOutputBinding::Fresh {
                            axis: 0,
                            extent_output_index: 1,
                            dimension: length.clone(),
                        },
                        CustomCallRaggedOutputBinding::Consumed,
                        CustomCallRaggedOutputBinding::Fresh {
                            axis: 0,
                            extent_output_index: 1,
                            dimension: length.clone(),
                        },
                        CustomCallRaggedOutputBinding::Consumed,
                    ],
                )),
            cases = [{
                input_types = [],
                output_types = [
                    output_types[0].clone(),
                    output_types[1].clone(),
                    output_types[2].clone(),
                    output_types[3].clone(),
                ],
            }],
        );
        check_operation_type_inference!(
            operation = CustomCallOperation::new("ryft.test.shared_output_dimension", output_types)
                .with_ragged_contract(CustomCallRaggedContract::new(
                    Vec::new(),
                    vec![
                        CustomCallRaggedOutputBinding::Fresh {
                            axis: 0,
                            extent_output_index: 1,
                            dimension: length.clone(),
                        },
                        CustomCallRaggedOutputBinding::Consumed,
                        CustomCallRaggedOutputBinding::Fresh { axis: 0, extent_output_index: 3, dimension: length },
                        CustomCallRaggedOutputBinding::Consumed,
                    ],
                )),
            cases = [{
                input_types = [],
                error = "`custom_call` fresh ragged outputs 0 and 2 reuse dimension `length` with different extent \
                         outputs 1 and 3",
            }],
        );
    }

    #[test]
    fn test_custom_call_rename_type_identities() {
        // Renaming rewrites every declared identity (dynamic output axes, ragged input bindings, and fresh ragged
        // outputs) in both universes, and preserves every other field.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let output_length = DimensionVariable::new("output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let renamed_rows = DimensionVariable::new("renamed_rows", DimensionBounds::new(1, Some(5)).unwrap());
        let renamed_length = DimensionVariable::new("renamed_length", DimensionBounds::new(0, Some(5)).unwrap());
        let renamed_output_length =
            DimensionVariable::new("renamed_output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let operation_with =
            |rows: &DimensionVariable, length: &DimensionVariable, output_length: &DimensionVariable| {
                CustomCallOperation::new(
                    "ryft.test.renamed",
                    vec![
                        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())])),
                        ArrayType::new_static(DataType::F32, [4]),
                        ArrayType::scalar(DataType::I32),
                    ],
                )
                .with_attribute("scale", 2.0)
                .with_input_output_alias(0, 1)
                .unwrap()
                .with_effect_class(EffectClass::UnorderedIo)
                .with_batching(CustomCallBatching::BroadcastAll)
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new("data", 1, 0, 2, length.clone())],
                    vec![
                        CustomCallRaggedOutputBinding::Consumed,
                        CustomCallRaggedOutputBinding::Fresh {
                            axis: 0,
                            extent_output_index: 2,
                            dimension: output_length.clone(),
                        },
                        CustomCallRaggedOutputBinding::Consumed,
                    ],
                ))
            };
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(rows.clone(), renamed_rows.clone()).unwrap();
        renaming.insert(length.clone(), renamed_length.clone()).unwrap();
        renaming.insert(output_length.clone(), renamed_output_length.clone()).unwrap();
        let expected = operation_with(&renamed_rows, &renamed_length, &renamed_output_length);
        let operation = operation_with(&rows, &length, &output_length);
        assert_eq!(operation.rename_type_identities(&renaming), Ok(expected.clone()));
        assert_eq!(operation.rename_parent_type_identities(&renaming), Ok(expected));
        assert_eq!(operation.rename_type_identities(&TypeIdentityRenaming::new()), Ok(operation));
    }

    #[test]
    fn test_custom_call_interpretation() {
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])]);
        assert!(matches!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![1.0, 2.0]).unwrap()],
            ),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "the reference array backend cannot execute the foreign kernel `ryft.test.add_one`",
        ));
    }

    #[test]
    fn test_custom_call_interpretation_array_ir() {
        // The reference backend has no kernel registry, so this fixture executes an identity kernel over real arrays
        // to exercise the mixed rule's verification of dynamic output extents.
        /// Real array that supplies an executable identity foreign kernel for extent-validation tests.
        #[derive(Clone, Debug, PartialEq)]
        struct IdentityKernelArray(Array);

        impl Display for IdentityKernelArray {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.fmt(formatter)
            }
        }

        impl Parameter for IdentityKernelArray {}

        impl Typed for IdentityKernelArray {
            type Type = ArrayType;

            fn r#type(&self) -> Cow<'_, ArrayType> {
                self.0.r#type()
            }
        }

        impl Value for IdentityKernelArray {
            type DispatchDomain = EagerContext<Self>;
            type ExecutionDomain = EagerContext<Self>;

            fn dispatch_domain(&self) -> Self::DispatchDomain {
                EagerContext::new()
            }

            fn execution_domain(&self) -> Self::ExecutionDomain {
                EagerContext::new()
            }
        }

        impl CustomCall for IdentityKernelArray {
            fn custom_call<'o, I: IntoIterator<Item = &'o Self>>(
                _operation: &CustomCallOperation,
                inputs: I,
            ) -> Result<Vec<Self>, ProgramError> {
                let inputs = inputs.into_iter().cloned().collect::<Vec<_>>();
                assert_eq!(inputs.len(), 1);
                Ok(inputs)
            }
        }

        impl DimensionSize<usize> for IdentityKernelArray {
            fn dimension_size<A: Into<Axis>>(&self, axis: A) -> Result<usize, ProgramError> {
                self.0.dimension_size(axis)
            }
        }

        let length = DimensionType::new("length", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = CustomCallOperation::new(
            "identity",
            vec![ArrayType::new(DataType::F32, Shape::new(vec![length.to_dimension()]))],
        );
        let input = ArrayIrValue::Array(IdentityKernelArray(Array::vector(vec![1.0f32, 2.0]).unwrap()));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(length.clone(), 2).unwrap());
        assert_eq!(ArrayIrValue::custom_call(&operation, [&input, &extent]), Ok(vec![input.clone()]));
        let wrong_extent = ArrayIrValue::Dimension(DimensionValue::new(length, 3).unwrap());
        assert!(matches!(
            ArrayIrValue::custom_call(&operation, [&input, &wrong_extent]),
            Err(ProgramError::InvalidArgument { message })
                if message == "`custom_call` output 0 axis 0 has extent 2, but its explicit extent input is 3",
        ));
        let wrong_identity = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        assert!(matches!(
            ArrayIrValue::custom_call(&operation, [&input, &wrong_identity]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`custom_call` output-extent input defines dimension variable `2`, but the \
                               corresponding declared output axis refers to `length`",
        ));
    }

    #[test]
    fn test_custom_call_partial_evaluation() {
        // `check_operation_partial_evaluation!` requires executable primal semantics, which the reference backend
        // cannot provide for a foreign kernel, so this test checks the default rule explicitly.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |input: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                CustomCall::custom_call(
                    &CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])]),
                    [&input],
                )
            },
            ArrayType::new_static(DataType::F32, [2]),
        )
        .unwrap();
        let program = program.to_flat_program();
        let residual = indoc! {"
            lambda %0:f32[2] .
            let %1:f32[2] = custom_call [target=kernel] %0
            in (%1)
        "}
        .trim_end();
        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(ArrayType::new_static(DataType::F32, [2]))])
            .unwrap();
        assert_eq!(evaluation.program().to_string(), residual);
        assert!(evaluation.outputs()[0].is_unknown());

        // A pure known call stays residual when the eager backend cannot execute it, preserving it for a backend that
        // supports the foreign kernel.
        let known = program
            .partially_evaluate(&[PartialValue::Known(Array::vector(vec![1.0f32, 2.0]).unwrap())])
            .unwrap();
        assert_eq!(known.program().to_string(), residual);
        assert!(known.outputs()[0].is_unknown());
        assert!(matches!(
            known.interpret(&EagerContext::<Array, ArrayOperation<Array>>::new(), &[]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "the reference array backend cannot execute the foreign kernel `kernel`",
        ));

        // A known effectful call propagates the execution failure, and disabling effect folding keeps even a known,
        // result-free call observable in the residual program.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder
            .add_instruction(
                CustomCallOperation::new("record", Vec::new()).with_side_effect(),
                Vec::new(),
                Vec::new(),
                None,
            )
            .unwrap();
        let program = builder.build::<Vec<Array>, Vec<Array>>(Vec::new(), Vec::new(), Vec::new()).unwrap();
        assert!(matches!(
            program.partially_evaluate(&[]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "the reference array backend cannot execute the foreign kernel `record`",
        ));
        let evaluation = program
            .entry_region_ref()
            .partially_evaluate_in_context(&EagerContext::<Array, ArrayOperation<Array>>::new(), &[], false)
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda  .
                let () = custom_call [target=record, has_side_effect=true]
                in ()
            "}
            .trim_end(),
        );
        assert_eq!(evaluation.program().effects().classes(), EffectClasses::single(EffectClass::OrderedIo));
    }

    #[test]
    fn test_custom_call_batching() {
        // The default behavior rejects a mapped input, naming it and its batch axis.
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first = builder.add_input(ArrayType::new_static(DataType::F32, [2]));
        let second = builder.add_input(ArrayType::new_static(DataType::F32, [2]));
        let output = builder.add_instruction(operation, Vec::new(), vec![first, second], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            program.batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message
                    == "custom call `ryft.test.add_one` has no batching rule for input 1 mapped at batch axis 0; \
                        invoke a kernel that understands the batch axis, or select an explicit batching behavior \
                        with `CustomCallOperation::with_batching`",
        ));

        // A call whose inputs are all replicated is bound unchanged and reports replicated outputs, because the
        // region-free foreign kernel cannot observe the transform's axis.
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::replicated()]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:f32[2] = custom_call [target=ryft.test.add_one] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_array_ir() {
        // The mixed rule rejects mapped inputs by default as well.
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])]);
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let mapped = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()),
            BatchAxis::new(0),
        )
        .unwrap();
        assert!(matches!(
            operation.batch_in_parent(&context, &EmptyRegionDriver, &[mapped]),
            Err(BatchingError::UnsupportedOperation { message })
                if message
                    == "custom call `ryft.test.add_one` has no batching rule for input 0 mapped at batch axis 0; \
                        invoke a kernel that understands the batch axis, or select an explicit batching behavior \
                        with `CustomCallOperation::with_batching`",
        ));

        // The all-replicated shortcut keeps the trailing first-class output-extent input as an ordinary replicated
        // input of the unchanged call.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let operation = CustomCallOperation::new("ryft.test.dynamic", vec![output_type.clone()]);
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = trace.input(ArrayType::new_static(DataType::F32, [2]).into());
        let extent = trace.input(DimensionType::from(rows).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            trace.clone(),
            trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())),
        );
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(input)),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)),
        ];
        let [output] = context.bind(operation, Vec::new(), &inputs).unwrap().try_into().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(output_type));
        let output_id = output.into_batch().into_value().atom_id().unwrap();
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output_id],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:dimension<rows ∈ [1, 9)> .
                let %2:dimension<2> = const 2
                    %3:f32[rows] = custom_call [target=ryft.test.dynamic] %0 %1
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_rejected_without_type_queries() {
        // A dense call without a ragged contract must reject a mapped input without projecting any input type.
        /// Array value whose type queries are counted for the no-contract batching fast-path regression.
        #[derive(Clone, Debug, PartialEq)]
        struct TypeReadCountingArray {
            /// Array type returned by [`Typed::r#type`].
            r#type: ArrayType,

            /// Number of type queries since the last reset.
            type_read_count: Rc<Cell<usize>>,
        }

        impl Display for TypeReadCountingArray {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.r#type.fmt(formatter)
            }
        }

        impl Parameter for TypeReadCountingArray {}

        impl Typed for TypeReadCountingArray {
            type Type = ArrayType;

            fn r#type(&self) -> Cow<'_, Self::Type> {
                self.type_read_count.set(self.type_read_count.get() + 1);
                Cow::Borrowed(&self.r#type)
            }
        }

        impl Value for TypeReadCountingArray {
            type DispatchDomain = TypeReadCountingContext;
            type ExecutionDomain = TypeReadCountingContext;

            fn dispatch_domain(&self) -> Self::DispatchDomain {
                TypeReadCountingContext
            }

            fn execution_domain(&self) -> Self::ExecutionDomain {
                TypeReadCountingContext
            }
        }

        /// Minimal context used only to prove that rejected dense batching performs no input type projection.
        #[derive(Copy, Clone)]
        struct TypeReadCountingContext;

        impl Domain for TypeReadCountingContext {
            type Type = ArrayType;
            type Value = TypeReadCountingArray;
            type Constant = TypeReadCountingArray;
            type Operation = ArrayOperation<TypeReadCountingArray>;
        }

        impl Context for TypeReadCountingContext {
            fn lift(&self, constant: Self::Constant) -> Result<Self::Value, ProgramError> {
                Ok(constant)
            }

            fn bind<O: Into<Self::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
                &self,
                _operation: O,
                _driver: D,
                _inputs: &[Self::Value],
            ) -> Result<Vec<Self::Value>, ProgramError> {
                unreachable!("the rejected custom-call batching path must not bind an operation")
            }

            fn is_eager(&self) -> bool {
                false
            }

            fn provenance(&self) -> Provenance {
                Provenance::unknown()
            }

            fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, _origin: Provenance, function: F) -> R {
                function()
            }

            fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, _scope: ProvenanceScope, function: F) -> R {
                function()
            }
        }

        let type_read_count = Rc::new(Cell::new(0));
        let input = TypeReadCountingArray {
            r#type: ArrayType::new_static(DataType::F32, [2, 3]),
            type_read_count: Rc::clone(&type_read_count),
        };
        let input = ArrayBatch::new(input, BatchAxis::new(0)).unwrap();
        type_read_count.set(0);

        let operation = CustomCallOperation::new("ryft.test.dense", vec![ArrayType::new_static(DataType::F32, [2])]);
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(TypeReadCountingContext, 2);
        assert!(matches!(
            operation.batch(&context, &EmptyRegionDriver, &[input]),
            Err(BatchingError::UnsupportedOperation { message })
                if message
                    == "custom call `ryft.test.dense` has no batching rule for input 0 mapped at batch axis 0; \
                        invoke a kernel that understands the batch axis, or select an explicit batching behavior \
                        with `CustomCallOperation::with_batching`",
        ));
        assert_eq!(type_read_count.get(), 0);
    }

    #[test]
    fn test_custom_call_batching_undeclared_ragged_inputs() {
        // Calls without a ragged contract reject ragged inputs before binding anything, whatever their behavior.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 3, vec![1.0f32; 6]).unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1i32, 3]).unwrap(), length, vec![0])])
            .unwrap();
        let operation =
            CustomCallOperation::new("ryft.test.side_effect", vec![ArrayType::new_static(DataType::F32, [2])])
                .with_batching(CustomCallBatching::BroadcastAll)
                .with_side_effect();
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        assert!(matches!(
            operation.batch(&context, &EmptyRegionDriver, &[input]),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "custom call `ryft.test.side_effect` does not support bounded ragged dimension `length` \
                               on input 0",
        ));
    }

    #[test]
    fn test_custom_call_batching_array_ir_extent_inputs() {
        // Trailing output-extent inputs are validated as replicated dimensions before any array projection.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)]));
        let operation = CustomCallOperation::new("ryft.test.dynamic", vec![output_type]);
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let inputs = [
            ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())),
            ArrayIrBatch::replicated(ArrayIrValue::Array(Array::scalar(2i32).unwrap())),
        ];
        assert!(matches!(
            operation.batch_in_parent(&context, &EmptyRegionDriver, &inputs),
            Err(BatchingError::Type(TypeError::Invalid { message }))
                if message == "expected dimension type but got array type",
        ));
        assert!(matches!(
            operation.batch_in_parent(&context, &EmptyRegionDriver, &[]),
            Err(BatchingError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 0 })),
        ));
    }

    #[test]
    fn test_custom_call_batching_sequential() {
        // `Sequential` stages one `scan` whose body performs exactly one unbatched call: the mapped input is sliced one
        // row per iteration while the replicated input rides along as an invariant carry. A side-effecting kernel
        // therefore runs once per batch item, in iteration order.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])])
                        .with_side_effect()
                        .with_batching(CustomCallBatching::Sequential { unroll: None });
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2]), ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let (batched, output_axes) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(batched.effects().classes(), EffectClasses::single(EffectClass::OrderedIo));
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2], %1:f32[2] .
                let %2:f32[2], %3:f32[3, 2] = scan [carry_count=1, length=3, reverse=false] %1 %0 [
                    body={
                        lambda %0:i64[], %1:f32[2], %2:f32[2] .
                        let %3:f32[2] = custom_call [target=ryft.test.add_one, has_side_effect=true, \
                batching=sequential] %2 %1
                        in (%1, %3)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_array_ir_sequential_extent_carries() -> Result<(), ProgramError> {
        // The mixed rule threads replicated first-class output extents as leading invariant scan carries, so the body's
        // call declares exactly the per-item extents it was given, and it supports a dynamic mapped extent by
        // consuming it as the scan's trailing trip-count input.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let operation = CustomCallOperation::new("ryft.test.dynamic", vec![output_type])
            .with_batching(CustomCallBatching::Sequential { unroll: None });
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::from(batch.clone()).into());
        let mapped = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2)])).into(),
        );
        let extent = trace.input(DimensionType::from(rows).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)),
        ];
        let [output] = context.bind(operation, Vec::new(), &inputs)?.try_into().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        let output_id = output.into_batch().into_value().atom_id()?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output_id],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 2], %2:dimension<rows ∈ [1, 9)> .
                let %3:dimension<rows ∈ [1, 9)>, %4:f32[batch, rows] = scan [carry_count=1, length=batch, \
                reverse=false] %2 %1 %0 [
                    body={
                        lambda %0:i64[], %1:dimension<rows ∈ [1, 9)>, %2:f32[2] .
                        let %3:f32[rows] = custom_call [target=ryft.test.dynamic, batching=sequential] %2 %1
                        in (%1, %3)
                    },
                ]
                in (%4)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_sequential_effects() {
        // The scan inherits the declared effect class of the call it repeats.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |input: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.record", vec![ArrayType::new_static(DataType::F32, [2])])
                        .with_effect_class(EffectClass::UnorderedIo)
                        .with_batching(CustomCallBatching::Sequential { unroll: None });
                Ok(CustomCall::custom_call(&operation, [&input])?.remove(0))
            },
            ArrayType::new_static(DataType::F32, [2]),
        )
        .unwrap();
        let (batched, output_axes) = program
            .to_flat_program()
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(batched.effects().classes(), EffectClasses::single(EffectClass::UnorderedIo));
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2] .
                let %1:f32[3, 2] = scan [carry_count=0, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[2] .
                        let %2:f32[2] = custom_call [
                            target=ryft.test.record,
                            has_side_effect=true,
                            effect_class=unordered_io,
                            batching=sequential,
                        ] %1
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_sequential_unroll() {
        // The lowering-only unroll factor is forwarded to the staged `scan`, even when it does not divide the batch
        // extent (lowerings run the remaining iterations after the unrolled loop).
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])])
                        .with_batching(CustomCallBatching::Sequential { unroll: Some(2) });
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let (batched, _) = program
            .batched(4, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[4, 2] .
                let %1:f32[4, 2] = scan [carry_count=0, length=4, reverse=false, unroll=2] %0 [
                    body={
                        lambda %0:i64[], %1:f32[2] .
                        let %2:f32[2] = custom_call [target=ryft.test.add_one, batching=sequential(unroll=2)] %1
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        let (batched, _) = program
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2] .
                let %1:f32[3, 2] = scan [carry_count=0, length=3, reverse=false, unroll=2] %0 [
                    body={
                        lambda %0:i64[], %1:f32[2] .
                        let %2:f32[2] = custom_call [target=ryft.test.add_one, batching=sequential(unroll=2)] %1
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );

        // A zero factor is rejected by the scan contract.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])])
                        .with_batching(CustomCallBatching::Sequential { unroll: Some(0) });
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        assert!(matches!(
            program.batched(
                4,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`scan` unroll factor must be at least 1",
        ));
    }

    #[test]
    fn test_custom_call_batching_sequential_unrolled() {
        // Full unrolling stages the same scan as `Sequential`, with an unroll factor equal to the static batch extent.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])])
                        .with_batching(CustomCallBatching::SequentialUnrolled);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let (batched, _) = program
            .batched(4, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[4, 2] .
                let %1:f32[4, 2] = scan [carry_count=0, length=4, reverse=false, unroll=4] %0 [
                    body={
                        lambda %0:i64[], %1:f32[2] .
                        let %2:f32[2] = custom_call [target=ryft.test.add_one, batching=sequential_unrolled] %1
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_array_ir_sequential_unrolled() -> Result<(), ProgramError> {
        // A dynamic batch extent cannot be fully unrolled.
        let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
            .with_batching(CustomCallBatching::SequentialUnrolled);
        let error = batch_array_ir_custom_call(operation).unwrap_err();
        assert!(matches!(
            error.downcast_custom::<BatchingError>(),
            Some(BatchingError::UnsupportedOperation { message })
                if message == "custom call `kernel` batching `sequential_unrolled` requires a statically known batch \
                               extent but the mapped axis has dynamic extent `batch`",
        ));

        // A static batch extent with a trailing output extent stays on the mixed rule and unrolls completely.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let operation = CustomCallOperation::new("ryft.test.dynamic", vec![output_type])
            .with_batching(CustomCallBatching::SequentialUnrolled);
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let [batch_extent] = trace
            .bind(ConstantOperation::new(DimensionValue::constant(4).unwrap()), Vec::new(), &[])?
            .try_into()
            .unwrap();
        let mapped = trace.input(ArrayType::new_static(DataType::F32, [4, 2]).into());
        let extent = trace.input(DimensionType::from(rows).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)),
        ];
        let [output] = context.bind(operation, Vec::new(), &inputs)?.try_into().unwrap();
        let output_id = output.into_batch().into_value().atom_id()?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output_id],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %1:f32[4, 2], %2:dimension<rows ∈ [1, 9)> .
                let %0:dimension<4> = constant [value=4]
                    %3:dimension<rows ∈ [1, 9)>, %4:f32[4, rows] = scan [carry_count=1, length=4, reverse=false, \
                unroll=4] %2 %1 [
                        body={
                            lambda %0:i64[], %1:dimension<rows ∈ [1, 9)>, %2:f32[2] .
                            let %3:f32[rows] = custom_call [target=ryft.test.dynamic, batching=sequential_unrolled] \
                %2 %1
                            in (%1, %3)
                        },
                    ]
                in (%4)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_sequential_input_output_aliases() {
        // `Sequential` preserves aliases per iteration, where the call sees the original per-item types.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])])
                        .with_input_output_alias(0, 0)?
                        .with_batching(CustomCallBatching::Sequential { unroll: None });
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let (batched, output_axes) = program
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2] .
                let %1:f32[3, 2] = scan [carry_count=0, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[2] .
                        let %2:f32[2] = custom_call [target=ryft.test.add_one, input_output_alias=0->0, \
                batching=sequential] %1
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_sequential_aliased_layouts() -> Result<(), ProgramError> {
        // Scan slices have unspecified storage, so the body restores the declared layout of an aliased input.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = trace.input(ArrayType::new_static(DataType::F32, [2, 2]));
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 2);
        let output_type = ArrayType::new_static(DataType::F32, [2])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])));
        let operation = CustomCallOperation::new("ryft.test.aliased_layout", vec![output_type])
            .with_input_output_alias(0, 0)?
            .with_batching(CustomCallBatching::Sequential { unroll: None });
        let (outputs, evidence) = operation
            .batch(&context, &EmptyRegionDriver, &[ArrayBatch::new(input, BatchAxis::new(0))?])?
            .into_parts();
        assert!(evidence.is_empty());
        let output_id = outputs[0].value().atom_id()?;
        let program = trace.builder().borrow().clone().build::<Vec<Array>, Vec<Array>>(
            vec![output_id],
            vec![Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 2] .
                let %1:f32[2, 2] = scan [carry_count=0, length=2, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[2] .
                        let %2:f32[2][layout=tiled{0:T(2)}] = broadcast [output_type=f32[2][layout=tiled{0:T(2)}], \
                output_axes=[0]] %1
                            %3:f32[2][layout=tiled{0:T(2)}] = custom_call [target=ryft.test.aliased_layout, \
                input_output_alias=0->0, batching=sequential] %2
                        in (%3)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_array_ir_sequential_aliased_layouts() -> Result<(), ProgramError> {
        // The mixed body resolves the slice's dimensions so that the relayout preserves dynamic geometry exactly.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(4))?);
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = trace.input(DimensionType::from(batch.clone()).into());
        let input = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2)])).into(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), extent);
        let output_type = ArrayType::new_static(DataType::F32, [2])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])));
        let operation = CustomCallOperation::new("ryft.test.aliased_layout", vec![output_type])
            .with_input_output_alias(0, 0)?
            .with_batching(CustomCallBatching::Sequential { unroll: None });
        let (outputs, evidence) = operation
            .batch_in_parent(&context, &EmptyRegionDriver, &[ArrayIrBatch::new(input, BatchAxis::new(0))?])?
            .into_parts();
        assert!(evidence.is_empty());
        let output_id = outputs[0].value().atom_id()?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output_id],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 4)>, %1:f32[batch, 2] .
                let %2:f32[batch, 2] = scan [carry_count=0, length=batch, reverse=false] %1 %0 [
                    body={
                        lambda %0:i64[], %1:f32[2] .
                        let %2:dimension<2> = dimension_size [axis=0] %1
                            %3:f32[2][layout=tiled{0:T(2)}] = broadcast [output_axes=[0], output_layout=tiled{0:T(2)}] \
                %1 %2
                            %4:f32[2][layout=tiled{0:T(2)}] = custom_call [target=ryft.test.aliased_layout, \
                input_output_alias=0->0, batching=sequential] %3
                        in (%4)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_sequential_input_layouts() {
        // The scan body performs the unbatched call, so declared input layouts stay unchanged.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
                    .with_input_layouts([
                        Some(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()))),
                        Some(Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()))),
                    ])
                    .with_batching(CustomCallBatching::Sequential { unroll: None });
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2, 3]), ArrayType::new_static(DataType::F32, [2, 3])],
        )
        .unwrap();
        let (batched, _) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2, 3], %1:f32[2, 3] .
                let %2:f32[2, 3], %3:f32[3, 2] = scan [carry_count=1, length=3, reverse=false] %1 %0 [
                    body={
                        lambda %0:i64[], %1:f32[2, 3], %2:f32[2, 3] .
                        let %3:f32[2] = custom_call [target=kernel, input_layouts=[tiled{0,1}, tiled{1,0}], \
                batching=sequential] %2 %1
                        in (%1, %3)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_array_ir_sequential_input_layouts() -> Result<(), ProgramError> {
        // The scan body performs the unbatched call, so declared input layouts stay unchanged.
        let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
            .with_input_layouts([
                Some(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()))),
                Some(Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()))),
            ])
            .with_batching(CustomCallBatching::Sequential { unroll: None });
        assert_eq!(
            batch_array_ir_custom_call(operation)?,
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 2, 3], %2:f32[2, 3] .
                let %3:f32[2, 3], %4:f32[batch, 2] = scan [carry_count=1, length=batch, reverse=false] %2 %1 %0 [
                    body={
                        lambda %0:i64[], %1:f32[2, 3], %2:f32[2, 3] .
                        let %3:f32[2] = custom_call [target=kernel, input_layouts=[tiled{0,1}, tiled{1,0}], \
                batching=sequential] %2 %1
                        in (%1, %3)
                    },
                ]
                in (%4)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_broadcast_all() {
        // `BroadcastAll` materializes every input on the batch axis and rebinds exactly one call whose declared
        // outputs gain the same leading extent. Output 0 aliases input 0 and therefore takes that input's packed type,
        // while output 1 keeps its declared tiled layout with every logical index shifted by one and the inserted
        // batch axis as its most major dimension.
        let column_major = ArrayType::new_static(DataType::F32, [2, 3])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new())));
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new(
                    "ryft.test.scaled_add",
                    vec![ArrayType::new_static(DataType::F32, [2]), column_major.clone()],
                )
                .with_input_output_alias(0, 0)?
                .with_batching(CustomCallBatching::BroadcastAll);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2]), ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let (batched, output_axes) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2], %1:f32[2] .
                let %2:f32[3, 2] = broadcast [output_type=f32[3, 2], output_axes=[1]] %1
                    %3:f32[3, 2], %4:f32[3, 2, 3][layout=tiled{1,2,0}] = custom_call [target=ryft.test.scaled_add, \
                input_output_alias=0->0, batching=broadcast_all] %0 %2
                in (%3, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_array_ir_broadcast_all() -> Result<(), ProgramError> {
        // The mixed rule regroups the trailing extents so each output's new leading dynamic axis is grounded by the
        // transform's extent.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let operation = CustomCallOperation::new("ryft.test.dynamic", vec![output_type])
            .with_batching(CustomCallBatching::BroadcastAll);
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::from(batch.clone()).into());
        let mapped = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2)])).into(),
        );
        let extent = trace.input(DimensionType::from(rows).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)),
        ];
        let [output] = context.bind(operation, Vec::new(), &inputs)?.try_into().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        let output_id = output.into_batch().into_value().atom_id()?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output_id],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 2], %2:dimension<rows ∈ [1, 9)> .
                let %3:f32[batch, rows] = custom_call [target=ryft.test.dynamic, batching=broadcast_all] %1 %0 %2
                in (%3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_broadcast_all_side_effect() {
        // `BroadcastAll` executes a side-effecting kernel exactly once over batch-prefixed buffers, unlike
        // `Sequential`, which executes it once per batch item.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.record", vec![ArrayType::new_static(DataType::F32, [2])])
                        .with_side_effect()
                        .with_batching(CustomCallBatching::BroadcastAll);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let (batched, _) = program
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(batched.effects().classes(), EffectClasses::single(EffectClass::OrderedIo));
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2] .
                let %1:f32[3, 2] = custom_call [target=ryft.test.record, has_side_effect=true, \
                batching=broadcast_all] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_broadcast_all_input_output_aliases() {
        // `BroadcastAll` preserves aliases across full batch-prefixed buffers.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])])
                        .with_input_output_alias(0, 0)?
                        .with_batching(CustomCallBatching::BroadcastAll);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let (batched, output_axes) = program
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2] .
                let %1:f32[3, 2] = custom_call [target=ryft.test.add_one, input_output_alias=0->0, \
                batching=broadcast_all] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_broadcast_all_aliased_layouts() -> Result<(), ProgramError> {
        // Mapped and materialized replicated aliased inputs are both relaid out to the shifted output layout.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mapped = trace.input(ArrayType::new_static(DataType::F32, [2, 2]));
        let output_type = ArrayType::new_static(DataType::F32, [2])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])));
        let replicated = trace.input(output_type.clone());
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 2);
        let operation = CustomCallOperation::new("ryft.test.aliased_layout", vec![output_type.clone(), output_type])
            .with_input_output_alias(0, 0)?
            .with_input_output_alias(1, 1)?
            .with_batching(CustomCallBatching::BroadcastAll);
        let (outputs, evidence) = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayBatch::new(mapped, BatchAxis::new(0))?, ArrayBatch::replicated(replicated)],
            )?
            .into_parts();
        assert!(evidence.is_empty());
        let output_ids = outputs.iter().map(|output| output.value().atom_id()).collect::<Result<Vec<_>, _>>()?;
        let program = trace.builder().borrow().clone().build::<Vec<Array>, Vec<Array>>(
            output_ids,
            vec![Placeholder, Placeholder],
            vec![Placeholder, Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 2], %1:f32[2][layout=tiled{0:T(2)}] .
                let %2:f32[2, 2] = broadcast [output_type=f32[2, 2], output_axes=[1]] %1
                    %3:f32[2, 2][layout=tiled{1,0:T(2)}] = broadcast [output_type=f32[2, 2][layout=tiled{1,0:T(2)}], \
                output_axes=[0, 1]] %0
                    %4:f32[2, 2][layout=tiled{1,0:T(2)}] = broadcast [output_type=f32[2, 2][layout=tiled{1,0:T(2)}], \
                output_axes=[0, 1]] %2
                    %5:f32[2, 2][layout=tiled{1,0:T(2)}], %6:f32[2, 2][layout=tiled{1,0:T(2)}] = custom_call [
                        target=ryft.test.aliased_layout,
                        input_output_alias=0->0,
                        input_output_alias=1->1,
                        batching=broadcast_all,
                    ] %3 %4
                in (%5, %6)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_array_ir_broadcast_all_aliased_layouts() -> Result<(), ProgramError> {
        // The mixed rule relays out aliased inputs through dimension-valued identity broadcasts.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(4))?);
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = trace.input(DimensionType::from(batch.clone()).into());
        let input = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2)])).into(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), extent);
        let output_type = ArrayType::new_static(DataType::F32, [2])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])));
        let operation = CustomCallOperation::new("ryft.test.aliased_layout", vec![output_type])
            .with_input_output_alias(0, 0)?
            .with_batching(CustomCallBatching::BroadcastAll);
        let (outputs, evidence) = operation
            .batch_in_parent(&context, &EmptyRegionDriver, &[ArrayIrBatch::new(input, BatchAxis::new(0))?])?
            .into_parts();
        assert!(evidence.is_empty());
        let output_id = outputs[0].value().atom_id()?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output_id],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 4)>, %1:f32[batch, 2] .
                let %2:dimension<2> = constant [value=2]
                    %3:f32[batch, 2][layout=tiled{1,0:T(2)}] = broadcast [output_axes=[0, 1], \
                output_layout=tiled{1,0:T(2)}] %1 %0 %2
                    %4:f32[batch, 2][layout=tiled{1,0:T(2)}] = custom_call [target=ryft.test.aliased_layout, \
                input_output_alias=0->0, batching=broadcast_all] %3 %0
                in (%4)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_broadcast_all_input_layouts() {
        // Every input gains the leading batch axis, so every declared layout shifts to keep it most major.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
                    .with_input_layouts([
                        Some(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()))),
                        Some(Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()))),
                    ])
                    .with_batching(CustomCallBatching::BroadcastAll);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2, 3]), ArrayType::new_static(DataType::F32, [2, 3])],
        )
        .unwrap();
        let (batched, _) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2, 3], %1:f32[2, 3] .
                let %2:f32[3, 2, 3] = broadcast [output_type=f32[3, 2, 3], output_axes=[1, 2]] %1
                    %3:f32[3, 2] = custom_call [
                        target=kernel,
                        input_layouts=[tiled{1,2,0}, tiled{2,1,0}],
                        batching=broadcast_all,
                    ] %0 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_array_ir_broadcast_all_input_layouts() -> Result<(), ProgramError> {
        // Every input gains the leading batch axis, including the replicated one, so every declared layout shifts.
        let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
            .with_input_layouts([
                Some(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()))),
                Some(Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()))),
            ])
            .with_batching(CustomCallBatching::BroadcastAll);
        assert_eq!(
            batch_array_ir_custom_call(operation)?,
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 2, 3], %2:f32[2, 3] .
                let %3:dimension<2> = constant [value=2]
                    %4:dimension<3> = constant [value=3]
                    %5:f32[batch, 2, 3] = broadcast [output_axes=[1, 2]] %2 %0 %3 %4
                    %6:f32[batch, 2] = custom_call [
                        target=kernel,
                        input_layouts=[tiled{1,2,0}, tiled{2,1,0}],
                        batching=broadcast_all,
                    ] %1 %5 %0
                in (%6)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_broadcast_all_strided_input_layouts() {
        // A strided input layout cannot gain a batch axis, because its byte stride is not derivable from the layout.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
                    .with_input_layouts([Some(Layout::Strided(StridedLayout::new(vec![4])))])
                    .with_batching(CustomCallBatching::BroadcastAll);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        assert!(matches!(
            program.batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "custom call `kernel` cannot batch input 0 because its strided layout `strided{4}` does \
                               not determine the byte stride of the inserted batch axis",
        ));
    }

    #[test]
    fn test_custom_call_batching_array_ir_broadcast_all_strided_input_layouts() {
        let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
            .with_input_layouts([Some(Layout::Strided(StridedLayout::new(vec![12, 4]))), None])
            .with_batching(CustomCallBatching::BroadcastAll);
        let error = batch_array_ir_custom_call(operation).unwrap_err();
        assert!(matches!(
            error.downcast_custom::<BatchingError>(),
            Some(BatchingError::UnsupportedOperation { message })
                if message == "custom call `kernel` cannot batch input 0 because its strided layout `strided{12,4}` \
                               does not determine the byte stride of the inserted batch axis",
        ));
    }

    #[test]
    fn test_custom_call_batching_broadcast_all_strided_output_layouts() {
        // A strided output layout cannot be shifted onto a batched declaration either.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let strided =
                    ArrayType::new_static(DataType::F32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
                let operation = CustomCallOperation::new("ryft.test.add_one", vec![strided])
                    .with_batching(CustomCallBatching::BroadcastAll);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        assert!(matches!(
            program.batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "custom call `ryft.test.add_one` cannot batch output 0 because its strided layout \
                               `strided{4}` does not determine the byte stride of the inserted batch axis",
        ));
    }

    #[test]
    fn test_custom_call_batching_broadcast_all_nested() {
        // The inner rule stages its rewritten instruction through the parent context, which is itself a batching
        // context, so the outer level batches that instruction structurally.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])])
                        .with_batching(CustomCallBatching::BroadcastAll);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let (inner, _) = program
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        let (outer, output_axes) = inner
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            outer.to_string(),
            indoc! {"
                lambda %0:f32[2, 3, 2] .
                let %1:f32[2, 3, 2] = custom_call [target=ryft.test.add_one, batching=broadcast_all] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_expand_dimensions() {
        // Mapped inputs move to batch axis 0 and replicated inputs gain a singleton leading axis.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.vectorized", vec![ArrayType::new_static(DataType::F32, [2])])
                        .with_input_output_alias(0, 0)?
                        .with_batching(CustomCallBatching::ExpandDimensions);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2]), ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let (batched, output_axes) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(1), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:f32[2] .
                let %2:f32[3, 2] = transpose [permutation=[1, 0]] %0
                    %3:f32[1, 2] = broadcast [output_type=f32[1, 2], output_axes=[1]] %1
                    %4:f32[3, 2] = custom_call [target=ryft.test.vectorized, input_output_alias=0->0, \
                batching=expand_dimensions] %2 %3
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_array_ir_expand_dimensions() -> Result<(), ProgramError> {
        // Singleton expansion preserves each replicated array's own extents rather than using the mapped extent.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let operation = CustomCallOperation::new("ryft.test.dynamic", vec![output_type])
            .with_batching(CustomCallBatching::ExpandDimensions);
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::from(batch.clone()).into());
        let mapped = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2)])).into(),
        );
        let invariant = trace.input(ArrayType::new_static(DataType::F32, [2]).into());
        let extent = trace.input(DimensionType::from(rows).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(invariant)),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)),
        ];
        let [output] = context.bind(operation, Vec::new(), &inputs)?.try_into().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        let output_id = output.into_batch().into_value().atom_id()?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output_id],
            vec![Placeholder; 4],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 2], %2:f32[2], %3:dimension<rows ∈ [1, 9)> .
                let %4:dimension<1> = constant [value=1]
                    %5:dimension<2> = constant [value=2]
                    %6:f32[1, 2] = broadcast [output_axes=[1]] %2 %4 %5
                    %7:f32[batch, rows] = custom_call [target=ryft.test.dynamic, batching=expand_dimensions] %1 %6 %0 %3
                in (%7)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_expand_dimensions_input_layouts() {
        // Replicated inputs gain a singleton leading axis, so their declared layouts shift like those of mapped ones.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
                    .with_input_layouts([
                        Some(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()))),
                        Some(Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()))),
                    ])
                    .with_batching(CustomCallBatching::ExpandDimensions);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2, 3]), ArrayType::new_static(DataType::F32, [2, 3])],
        )
        .unwrap();
        let (batched, _) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2, 3], %1:f32[2, 3] .
                let %2:f32[1, 2, 3] = broadcast [output_type=f32[1, 2, 3], output_axes=[1, 2]] %1
                    %3:f32[3, 2] = custom_call [
                        target=kernel,
                        input_layouts=[tiled{1,2,0}, tiled{2,1,0}],
                        batching=expand_dimensions,
                    ] %0 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_array_ir_expand_dimensions_input_layouts() -> Result<(), ProgramError> {
        // The replicated input gains a singleton leading axis, so its declared layout shifts like the mapped one.
        let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
            .with_input_layouts([
                Some(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()))),
                Some(Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()))),
            ])
            .with_batching(CustomCallBatching::ExpandDimensions);
        assert_eq!(
            batch_array_ir_custom_call(operation)?,
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 2, 3], %2:f32[2, 3] .
                let %3:dimension<1> = constant [value=1]
                    %4:dimension<2> = constant [value=2]
                    %5:dimension<3> = constant [value=3]
                    %6:f32[1, 2, 3] = broadcast [output_axes=[1, 2]] %2 %3 %4 %5
                    %7:f32[batch, 2] = custom_call [
                        target=kernel,
                        input_layouts=[tiled{1,2,0}, tiled{2,1,0}],
                        batching=expand_dimensions,
                    ] %1 %6 %0
                in (%7)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_expand_dimensions_invariant_aliases() {
        // A singleton-expanded replicated input cannot alias an output that gains the full batch extent.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mapped = trace.input(ArrayType::new_static(DataType::F32, [3, 2]));
        let invariant = trace.input(ArrayType::new_static(DataType::F32, [2]));
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace, 3);
        let operation =
            CustomCallOperation::new("ryft.test.vectorized", vec![ArrayType::new_static(DataType::F32, [2])])
                .with_input_output_alias(1, 0)
                .unwrap()
                .with_batching(CustomCallBatching::ExpandDimensions);
        assert!(matches!(
            operation.batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayBatch::new(mapped, BatchAxis::new(0)).unwrap(), ArrayBatch::replicated(invariant)],
            ),
            Err(BatchingError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`custom_call` alias `1->0` requires matching input and output types but input 1 has \
                               type `f32[1, 2]` and output 0 has type `f32[3, 2]`",
        ));
    }

    #[test]
    fn test_custom_call_batching_expand_dimensions_nested() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.vectorized", vec![ArrayType::new_static(DataType::F32, [2])])
                        .with_batching(CustomCallBatching::ExpandDimensions);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let (inner, _) = program
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        let (outer, output_axes) = inner
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            outer.to_string(),
            indoc! {"
                lambda %0:f32[2, 3, 2] .
                let %1:f32[2, 3, 2] = custom_call [target=ryft.test.vectorized, batching=expand_dimensions] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_vectorized() {
        // Mapped inputs move to batch axis 0 and replicated inputs keep their original geometry.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.vectorized", vec![ArrayType::new_static(DataType::F32, [2])])
                        .with_input_output_alias(0, 0)?
                        .with_batching(CustomCallBatching::Vectorized);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2]), ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let (batched, output_axes) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(1), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:f32[2] .
                let %2:f32[3, 2] = transpose [permutation=[1, 0]] %0
                    %3:f32[3, 2] = custom_call [target=ryft.test.vectorized, input_output_alias=0->0, \
                batching=vectorized] %2 %1
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_array_ir_vectorized() -> Result<(), ProgramError> {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let operation = CustomCallOperation::new("ryft.test.dynamic", vec![output_type])
            .with_batching(CustomCallBatching::Vectorized);
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::from(batch.clone()).into());
        let mapped = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2)])).into(),
        );
        let invariant = trace.input(ArrayType::new_static(DataType::F32, [2]).into());
        let extent = trace.input(DimensionType::from(rows).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(invariant)),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)),
        ];
        let [output] = context.bind(operation, Vec::new(), &inputs)?.try_into().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        let output_id = output.into_batch().into_value().atom_id()?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output_id],
            vec![Placeholder; 4],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 2], %2:f32[2], %3:dimension<rows ∈ [1, 9)> .
                let %4:f32[batch, rows] = custom_call [target=ryft.test.dynamic, batching=vectorized] %1 %2 %0 %3
                in (%4)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_vectorized_input_layouts() {
        // Replicated inputs keep their geometry, so only the declared layouts of mapped inputs shift.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
                    .with_input_layouts([
                        Some(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()))),
                        Some(Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()))),
                    ])
                    .with_batching(CustomCallBatching::Vectorized);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2, 3]), ArrayType::new_static(DataType::F32, [2, 3])],
        )
        .unwrap();
        let (batched, _) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2, 3], %1:f32[2, 3] .
                let %2:f32[3, 2] = custom_call [target=kernel, input_layouts=[tiled{1,2,0}, tiled{1,0}], \
                batching=vectorized] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_array_ir_vectorized_input_layouts() -> Result<(), ProgramError> {
        // Only the mapped input gains the leading batch axis, so the replicated input keeps its declared layout.
        let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])])
            .with_input_layouts([
                Some(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()))),
                Some(Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()))),
            ])
            .with_batching(CustomCallBatching::Vectorized);
        assert_eq!(
            batch_array_ir_custom_call(operation)?,
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 2, 3], %2:f32[2, 3] .
                let %3:f32[batch, 2] = custom_call [target=kernel, input_layouts=[tiled{1,2,0}, tiled{1,0}], \
                batching=vectorized] %1 %2 %0
                in (%3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_vectorized_invariant_aliases() {
        // An unchanged replicated input cannot alias an output that gains the batch extent.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mapped = trace.input(ArrayType::new_static(DataType::F32, [3, 2]));
        let invariant = trace.input(ArrayType::new_static(DataType::F32, [2]));
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace, 3);
        let operation =
            CustomCallOperation::new("ryft.test.vectorized", vec![ArrayType::new_static(DataType::F32, [2])])
                .with_input_output_alias(1, 0)
                .unwrap()
                .with_batching(CustomCallBatching::Vectorized);
        assert!(matches!(
            operation.batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayBatch::new(mapped, BatchAxis::new(0)).unwrap(), ArrayBatch::replicated(invariant)],
            ),
            Err(BatchingError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`custom_call` alias `1->0` requires matching input and output types but input 1 has \
                               type `f32[2]` and output 0 has type `f32[3, 2]`",
        ));
    }

    #[test]
    fn test_custom_call_batching_vectorized_nested() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.vectorized", vec![ArrayType::new_static(DataType::F32, [2])])
                        .with_batching(CustomCallBatching::Vectorized);
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let (inner, _) = program
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        let (outer, output_axes) = inner
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            outer.to_string(),
            indoc! {"
                lambda %0:f32[2, 3, 2] .
                let %1:f32[2, 3, 2] = custom_call [target=ryft.test.vectorized, batching=vectorized] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_ragged_contract_sequential() -> Result<(), ProgramError> {
        // The scan body sees one unbatched slice at a time, so the contract keeps its axes and records the discharge.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 4]));
        let extents = trace.input(ArrayType::new_static(DataType::I32, [2]));
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 2);
        let data = ArrayBatch::new(packed, BatchAxis::new(0))?.with_ragged_axes(vec![RaggedAxis::new(
            1,
            extents.clone(),
            length.clone(),
            vec![0],
        )])?;
        let extent_input = ArrayBatch::new(extents.clone(), BatchAxis::new(0))?;
        let operation = CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [4])])
            .with_batching(CustomCallBatching::Sequential { unroll: None })
            .with_ragged_contract(preserved_ragged_contract(length.clone()));
        let (outputs, evidence) = operation.batch(&context, &EmptyRegionDriver, &[data, extent_input])?.into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].ragged_axes(), &[RaggedAxis::new(1, extents, length, vec![0])]);
        let program = trace.builder().borrow().clone().build::<Vec<Array>, Vec<Array>>(
            vec![outputs[0].value().atom_id()?],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4], %1:i32[2] .
                let %2:f32[2, 4] = scan [carry_count=0, length=2, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[4], %2:i32[] .
                        let %3:f32[4] = custom_call [
                            target=ryft.test.ragged,
                            batching=sequential,
                            ragged_contract={inputs=[data:input(0)@0<=input(1):length], \
                outputs=[preserve(data)@0], ragged_discharged=true},
                        ] %1 %2
                        in (%3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_array_ir_ragged_contract_sequential() -> Result<(), ProgramError> {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let batch_size = DimensionVariable::new("batch_size", DimensionBounds::new(1, Some(5)).unwrap());
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let axis_extent = trace.input(DimensionType::from(batch_size.clone()).into());
        let packed = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(batch_size.clone()), Dimension::Static(4)]),
            )
            .into(),
        );
        let extents =
            trace.input(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(batch_size)])).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), axis_extent);
        let data = ArrayIrBatch::new(packed, BatchAxis::new(0))?.with_ragged_axes(vec![RaggedAxis::new(
            1,
            extents.clone(),
            length.clone(),
            vec![0],
        )])?;
        let extent_input = ArrayIrBatch::new(extents, BatchAxis::new(0))?;
        let operation = CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [4])])
            .with_batching(CustomCallBatching::Sequential { unroll: None })
            .with_ragged_contract(preserved_ragged_contract(length));
        let (outputs, evidence) =
            operation.batch_in_parent(&context, &EmptyRegionDriver, &[data, extent_input])?.into_parts();
        assert!(evidence.is_empty());
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![outputs[0].value().atom_id()?],
            vec![Placeholder; 3],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch_size ∈ [1, 5)>, %1:f32[batch_size, 4], %2:i32[batch_size] .
                let %3:f32[batch_size, 4] = scan [carry_count=0, length=batch_size, reverse=false] %1 %2 %0 [
                    body={
                        lambda %0:i64[], %1:f32[4], %2:i32[] .
                        let %3:f32[4] = custom_call [
                            target=ryft.test.ragged,
                            batching=sequential,
                            ragged_contract={inputs=[data:input(0)@0<=input(1):length], \
                outputs=[preserve(data)@0], ragged_discharged=true},
                        ] %1 %2
                        in (%3)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_ragged_contract_broadcast_all() -> Result<(), ProgramError> {
        // Single-call batching shifts every bound axis behind the new batch axis and records the discharge.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 4]));
        let extents = trace.input(ArrayType::new_static(DataType::I32, [2]));
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 2);
        let data = ArrayBatch::new(packed, BatchAxis::new(0))?.with_ragged_axes(vec![RaggedAxis::new(
            1,
            extents.clone(),
            length.clone(),
            vec![0],
        )])?;
        let extent_input = ArrayBatch::new(extents.clone(), BatchAxis::new(0))?;
        let operation = CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [4])])
            .with_batching(CustomCallBatching::BroadcastAll)
            .with_ragged_contract(preserved_ragged_contract(length.clone()));
        let (outputs, evidence) = operation.batch(&context, &EmptyRegionDriver, &[data, extent_input])?.into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].ragged_axes(), &[RaggedAxis::new(1, extents, length, vec![0])]);
        let program = trace.builder().borrow().clone().build::<Vec<Array>, Vec<Array>>(
            vec![outputs[0].value().atom_id()?],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4], %1:i32[2] .
                let %2:f32[2, 4] = custom_call [
                    target=ryft.test.ragged,
                    batching=broadcast_all,
                    ragged_contract={inputs=[data:input(0)@1<=input(1):length], \
                outputs=[preserve(data)@1], batch_prefix_count=1, ragged_discharged=true},
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_array_ir_ragged_contract_broadcast_all() -> Result<(), ProgramError> {
        // A static mapped extent delegates to the projected homogeneous rule, which must keep the contract.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 4]).into());
        let extents = trace.input(ArrayType::new_static(DataType::I32, [2]).into());
        let axis_extent = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), axis_extent);
        let data = ArrayIrBatch::new(packed, BatchAxis::new(0))?.with_ragged_axes(vec![RaggedAxis::new(
            1,
            extents.clone(),
            length.clone(),
            vec![0],
        )])?;
        let extent_input = ArrayIrBatch::new(extents.clone(), BatchAxis::new(0))?;
        let operation = CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [4])])
            .with_batching(CustomCallBatching::BroadcastAll)
            .with_ragged_contract(preserved_ragged_contract(length.clone()));
        let (outputs, evidence) =
            operation.batch_in_parent(&context, &EmptyRegionDriver, &[data, extent_input])?.into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].ragged_axes(), &[RaggedAxis::new(1, extents, length, vec![0])]);
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![outputs[0].value().atom_id()?],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4], %1:i32[2] .
                let %2:dimension<2> = const 2
                    %3:f32[2, 4] = custom_call [
                        target=ryft.test.ragged,
                        batching=broadcast_all,
                        ragged_contract={inputs=[data:input(0)@1<=input(1):length], \
                outputs=[preserve(data)@1], batch_prefix_count=1, ragged_discharged=true},
                    ] %0 %1
                in (%3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_ragged_contract_dense_prefix() -> Result<(), ProgramError> {
        // A first ragged discharge after a dense batching level aligns the extent input to the complete prefix.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 3, 4]));
        let extents = trace.input(ArrayType::new_static(DataType::I32, [3, 2]));
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 2);
        let data = ArrayBatch::new(packed, BatchAxis::new(0))?.with_ragged_axes(vec![RaggedAxis::new(
            2,
            extents.clone(),
            length.clone(),
            vec![1, 0],
        )])?;
        let extent_input = ArrayBatch::new(extents, BatchAxis::new(1))?;
        let operation =
            CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [3, 4])])
                .with_batching(CustomCallBatching::BroadcastAll)
                .with_ragged_contract(preserved_ragged_contract(length.clone()).batch_prefixed(false));
        let (outputs, evidence) = operation.batch(&context, &EmptyRegionDriver, &[data, extent_input])?.into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].ragged_axes().len(), 1);
        assert_eq!(outputs[0].ragged_axes()[0].axis(), 2);
        assert_eq!(outputs[0].ragged_axes()[0].extent_axes(), &[0, 1]);
        assert_eq!(outputs[0].ragged_axes()[0].dimension(), &length);
        assert_eq!(
            outputs[0].ragged_axes()[0].extents().r#type().into_owned(),
            ArrayType::new_static(DataType::I32, [2, 3]),
        );
        let program = trace.builder().borrow().clone().build::<Vec<Array>, Vec<Array>>(
            vec![outputs[0].value().atom_id()?],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3, 4], %1:i32[3, 2] .
                let %2:i32[2, 3] = transpose [permutation=[1, 0]] %1
                    %3:f32[2, 3, 4] = custom_call [
                        target=ryft.test.ragged,
                        batching=broadcast_all,
                        ragged_contract={inputs=[data:input(0)@2<=input(1):length], \
                outputs=[preserve(data)@2], batch_prefix_count=2, ragged_discharged=true},
                    ] %0 %2
                in (%3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_array_ir_ragged_contract_dense_prefix() -> Result<(), ProgramError> {
        // The mixed path performs the same alignment and must attach the aligned extent value as well.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 3, 4]).into());
        let extents = trace.input(ArrayType::new_static(DataType::I32, [3, 2]).into());
        let axis_extent = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), axis_extent);
        let data = ArrayIrBatch::new(packed, BatchAxis::new(0))?.with_ragged_axes(vec![RaggedAxis::new(
            2,
            extents.clone(),
            length.clone(),
            vec![1, 0],
        )])?;
        let extent_input = ArrayIrBatch::new(extents, BatchAxis::new(1))?;
        let operation =
            CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [3, 4])])
                .with_batching(CustomCallBatching::BroadcastAll)
                .with_ragged_contract(preserved_ragged_contract(length).batch_prefixed(false));
        let (outputs, evidence) =
            operation.batch_in_parent(&context, &EmptyRegionDriver, &[data, extent_input])?.into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].ragged_axes()[0].axis(), 2);
        assert_eq!(outputs[0].ragged_axes()[0].extent_axes(), &[0, 1]);
        assert_eq!(
            outputs[0].ragged_axes()[0].extents().r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2, 3])),
        );
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![outputs[0].value().atom_id()?],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3, 4], %1:i32[3, 2] .
                let %2:dimension<2> = const 2
                    %3:i32[2, 3] = transpose [permutation=[1, 0]] %1
                    %4:f32[2, 3, 4] = custom_call [
                        target=ryft.test.ragged,
                        batching=broadcast_all,
                        ragged_contract={inputs=[data:input(0)@2<=input(1):length], \
                outputs=[preserve(data)@2], batch_prefix_count=2, ragged_discharged=true},
                    ] %0 %3
                in (%4)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_ragged_contract_nested_dense() {
        // Repeated dense batching accumulates the complete batch prefix in the contract.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let packed_type = ArrayType::new_static(DataType::F32, [4]);
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                    .with_batching(CustomCallBatching::BroadcastAll)
                    .with_ragged_contract(preserved_ragged_contract(length.clone()));
                CustomCall::custom_call(&operation, inputs.iter())
            },
            vec![packed_type.clone(), ArrayType::scalar(DataType::I32)],
        )
        .unwrap();
        let (inner, _) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        let (outer, output_axes) = inner
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            outer.to_string(),
            indoc! {"
                lambda %0:f32[2, 3, 4], %1:i32[2, 3] .
                let %2:f32[2, 3, 4] = custom_call [
                    target=ryft.test.ragged,
                    batching=broadcast_all,
                    ragged_contract={inputs=[data:input(0)@2<=input(1):length], \
                outputs=[preserve(data)@2], batch_prefix_count=2},
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batching_ragged_contract_fresh_outputs() -> Result<(), ProgramError> {
        // A fresh output takes its extents from the declared extent output, and a consumed input dimension is
        // reported as evidence.
        let input_length = DimensionVariable::new("input_length", DimensionBounds::new(0, Some(5)).unwrap());
        let output_length = DimensionVariable::new("output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 4]));
        let extents = trace.input(ArrayType::new_static(DataType::I32, [2]));
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 2);
        let data = ArrayBatch::new(packed, BatchAxis::new(0))?.with_ragged_axes(vec![RaggedAxis::new(
            1,
            extents.clone(),
            input_length.clone(),
            vec![0],
        )])?;
        let extent_input = ArrayBatch::new(extents, BatchAxis::new(0))?;
        let operation = CustomCallOperation::new(
            "ryft.test.fresh_ragged",
            vec![ArrayType::new_static(DataType::F32, [4]), ArrayType::scalar(DataType::I32)],
        )
        .with_batching(CustomCallBatching::BroadcastAll)
        .with_ragged_contract(CustomCallRaggedContract::new(
            vec![CustomCallRaggedInputBinding::new("data", 0, 0, 1, input_length.clone())],
            vec![
                CustomCallRaggedOutputBinding::Fresh {
                    axis: 0,
                    extent_output_index: 1,
                    dimension: output_length.clone(),
                },
                CustomCallRaggedOutputBinding::Consumed,
            ],
        ));
        let (outputs, evidence) = operation.batch(&context, &EmptyRegionDriver, &[data, extent_input])?.into_parts();
        assert_eq!(evidence, vec![input_length]);
        assert_eq!(outputs[0].ragged_axes(), &[RaggedAxis::new(1, outputs[1].value().clone(), output_length, vec![0])]);
        assert!(outputs[1].ragged_axes().is_empty());
        let output_ids = outputs.iter().map(|output| output.value().atom_id()).collect::<Result<Vec<_>, _>>()?;
        let program = trace.builder().borrow().clone().build::<Vec<Array>, Vec<Array>>(
            output_ids,
            vec![Placeholder, Placeholder],
            vec![Placeholder, Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4], %1:i32[2] .
                let %2:f32[2, 4], %3:i32[2] = custom_call [
                    target=ryft.test.fresh_ragged,
                    batching=broadcast_all,
                    ragged_contract={inputs=[data:input(0)@1<=input(1):input_length], \
                outputs=[fresh@1<=output(1):output_length, consume], batch_prefix_count=1, ragged_discharged=true},
                ] %0 %1
                in (%2, %3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_ragged_contract_fresh_outputs_without_ragged_inputs() -> Result<(), ProgramError> {
        // Fresh outputs are attached even when no input is ragged, so no input dimension is consumed.
        let output_length = DimensionVariable::new("output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = trace.input(ArrayType::new_static(DataType::F32, [2, 4]));
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 2);
        let operation = CustomCallOperation::new(
            "ryft.test.fresh_ragged",
            vec![ArrayType::new_static(DataType::F32, [4]), ArrayType::scalar(DataType::I32)],
        )
        .with_batching(CustomCallBatching::BroadcastAll)
        .with_ragged_contract(CustomCallRaggedContract::new(
            Vec::new(),
            vec![
                CustomCallRaggedOutputBinding::Fresh {
                    axis: 0,
                    extent_output_index: 1,
                    dimension: output_length.clone(),
                },
                CustomCallRaggedOutputBinding::Consumed,
            ],
        ));
        let (outputs, evidence) = operation
            .batch(&context, &EmptyRegionDriver, &[ArrayBatch::new(input, BatchAxis::new(0))?])?
            .into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs[0].ragged_axes(), &[RaggedAxis::new(1, outputs[1].value().clone(), output_length, vec![0])]);
        assert!(outputs[1].ragged_axes().is_empty());
        let output_ids = outputs.iter().map(|output| output.value().atom_id()).collect::<Result<Vec<_>, _>>()?;
        let program = trace.builder().borrow().clone().build::<Vec<Array>, Vec<Array>>(
            output_ids,
            vec![Placeholder],
            vec![Placeholder, Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4] .
                let %1:f32[2, 4], %2:i32[2] = custom_call [
                    target=ryft.test.fresh_ragged,
                    batching=broadcast_all,
                    ragged_contract={inputs=[], outputs=[fresh@1<=output(1):output_length, consume], \
                batch_prefix_count=1},
                ] %0
                in (%1, %2)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_ragged_contract_fresh_outputs_replicated() -> Result<(), ProgramError> {
        // An all-replicated call is bound unchanged, and its fresh output carries no batch-prefixed extent axes.
        let output_length = DimensionVariable::new("output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = trace.input(ArrayType::new_static(DataType::F32, [4]));
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 2);
        let operation = CustomCallOperation::new(
            "ryft.test.fresh_ragged",
            vec![ArrayType::new_static(DataType::F32, [4]), ArrayType::scalar(DataType::I32)],
        )
        .with_batching(CustomCallBatching::BroadcastAll)
        .with_ragged_contract(CustomCallRaggedContract::new(
            Vec::new(),
            vec![
                CustomCallRaggedOutputBinding::Fresh {
                    axis: 0,
                    extent_output_index: 1,
                    dimension: output_length.clone(),
                },
                CustomCallRaggedOutputBinding::Consumed,
            ],
        ));
        let (outputs, evidence) =
            operation.batch(&context, &EmptyRegionDriver, &[ArrayBatch::replicated(input)])?.into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(
            outputs[0].ragged_axes(),
            &[RaggedAxis::new(0, outputs[1].value().clone(), output_length, Vec::new())],
        );
        let output_ids = outputs.iter().map(|output| output.value().atom_id()).collect::<Result<Vec<_>, _>>()?;
        let program = trace.builder().borrow().clone().build::<Vec<Array>, Vec<Array>>(
            output_ids,
            vec![Placeholder],
            vec![Placeholder, Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4], %2:i32[] = custom_call [
                    target=ryft.test.fresh_ragged,
                    batching=broadcast_all,
                    ragged_contract={inputs=[], outputs=[fresh@0<=output(1):output_length, consume]},
                ] %0
                in (%1, %2)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_array_ir_ragged_contract_fresh_outputs_replicated() -> Result<(), ProgramError> {
        let output_length = DimensionVariable::new("output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let batch_size = DimensionVariable::new("batch_size", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = CustomCallOperation::new(
            "ryft.test.fresh_ragged",
            vec![ArrayType::new_static(DataType::F32, [4]), ArrayType::scalar(DataType::I32)],
        )
        .with_ragged_contract(CustomCallRaggedContract::new(
            Vec::new(),
            vec![
                CustomCallRaggedOutputBinding::Fresh {
                    axis: 0,
                    extent_output_index: 1,
                    dimension: output_length.clone(),
                },
                CustomCallRaggedOutputBinding::Consumed,
            ],
        ));
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = trace.input(ArrayType::new_static(DataType::F32, [4]).into());
        let axis_extent = trace.input(DimensionType::from(batch_size).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), axis_extent);
        let (outputs, evidence) = operation
            .batch_in_parent(&context, &EmptyRegionDriver, &[ArrayIrBatch::replicated(input)])?
            .into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(
            outputs[0].ragged_axes(),
            &[RaggedAxis::new(0, outputs[1].value().clone(), output_length, Vec::new())],
        );
        let output_ids = outputs.iter().map(|output| output.value().atom_id()).collect::<Result<Vec<_>, _>>()?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            output_ids,
            vec![Placeholder, Placeholder],
            vec![Placeholder, Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:dimension<batch_size ∈ [1, 5)> .
                let %2:f32[4], %3:i32[] = custom_call [
                    target=ryft.test.fresh_ragged,
                    ragged_contract={inputs=[], outputs=[fresh@0<=output(1):output_length, consume]},
                ] %0
                in (%2, %3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_array_ir_ragged_contract_fresh_outputs_nested_dense() -> Result<(), ProgramError> {
        // A fresh output of a call that was already batch-prefixed once keeps the complete extent-axis mapping.
        let output_length = DimensionVariable::new("output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let operation = CustomCallOperation::new(
            "ryft.test.fresh_ragged",
            vec![ArrayType::new_static(DataType::F32, [3, 4]), ArrayType::new_static(DataType::I32, [3])],
        )
        .with_batching(CustomCallBatching::BroadcastAll)
        .with_ragged_contract(
            CustomCallRaggedContract::new(
                Vec::new(),
                vec![
                    CustomCallRaggedOutputBinding::Fresh {
                        axis: 0,
                        extent_output_index: 1,
                        dimension: output_length.clone(),
                    },
                    CustomCallRaggedOutputBinding::Consumed,
                ],
            )
            .batch_prefixed(false),
        );
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let outer = DimensionVariable::new("outer", DimensionBounds::new(1, Some(5)).unwrap());
        let axis_extent = trace.input(DimensionType::from(outer.clone()).into());
        let input = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(outer), Dimension::Static(3), Dimension::Static(4)]),
            )
            .into(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), axis_extent);
        let (outputs, evidence) = operation
            .batch_in_parent(&context, &EmptyRegionDriver, &[ArrayIrBatch::new(input, BatchAxis::new(0))?])?
            .into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            outputs[0].ragged_axes(),
            &[RaggedAxis::new(2, outputs[1].value().clone(), output_length, vec![0, 1])],
        );
        let output_ids = outputs.iter().map(|output| output.value().atom_id()).collect::<Result<Vec<_>, _>>()?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            output_ids,
            vec![Placeholder, Placeholder],
            vec![Placeholder, Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<outer ∈ [1, 5)>, %1:f32[outer, 3, 4] .
                let %2:f32[outer, 3, 4], %3:i32[outer, 3] = custom_call [
                    target=ryft.test.fresh_ragged,
                    batching=broadcast_all,
                    ragged_contract={inputs=[], outputs=[fresh@2<=output(1):output_length, consume], \
                batch_prefix_count=2},
                ] %1 %0 %0
                in (%2, %3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_custom_call_batching_ragged_contract_extent_mismatch() {
        // The declared extent input must be the exact extent value carried by the ragged input.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 4]));
        let extents = trace.input(ArrayType::new_static(DataType::I32, [2]));
        let other_extents = trace.input(ArrayType::new_static(DataType::I32, [2]));
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace, 2);
        let data = ArrayBatch::new(packed, BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents, length.clone(), vec![0])])
            .unwrap();
        let extent_input = ArrayBatch::new(other_extents, BatchAxis::new(0)).unwrap();
        let operation = CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [4])])
            .with_batching(CustomCallBatching::BroadcastAll)
            .with_ragged_contract(preserved_ragged_contract(length));
        assert!(matches!(
            operation.batch(&context, &EmptyRegionDriver, &[data, extent_input]),
            Err(BatchingError::InvalidBatchMetadata { message })
                if message == "custom call `ryft.test.ragged` ragged input binding `data` requires input 1 to be \
                               the exact extent value carried by input 0",
        ));
    }

    #[test]
    fn test_custom_call_batching_ragged_contract_nested_ragged() {
        // A contract that already discharged one ragged level rejects another ragged input.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 4]));
        let extents = trace.input(ArrayType::new_static(DataType::I32, [2]));
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace, 2);
        let data = ArrayBatch::new(packed, BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), length.clone(), vec![0])])
            .unwrap();
        let extent_input = ArrayBatch::new(extents, BatchAxis::new(0)).unwrap();
        let operation = CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [4])])
            .with_batching(CustomCallBatching::BroadcastAll)
            .with_ragged_contract(preserved_ragged_contract(length).ragged_discharged());
        assert!(matches!(
            operation.batch(&context, &EmptyRegionDriver, &[data, extent_input]),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "custom call `ryft.test.ragged` does not support nested ragged batching",
        ));
    }

    #[test]
    fn test_custom_call_batching_ragged_contract_expand_dimensions_invariant_bindings() {
        // Singleton-expanded replicated inputs cannot carry the full batch extent that a ragged binding describes.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [4]));
        let extent = trace.input(ArrayType::scalar(DataType::I32));
        let mapped = trace.input(ArrayType::new_static(DataType::F32, [3, 2]));
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace, 3);
        let operation = CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [4])])
            .with_batching(CustomCallBatching::ExpandDimensions)
            .with_ragged_contract(preserved_ragged_contract(length));
        assert!(matches!(
            operation.batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayBatch::replicated(packed),
                    ArrayBatch::replicated(extent),
                    ArrayBatch::new(mapped, BatchAxis::new(0)).unwrap(),
                ],
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "custom call `ryft.test.ragged` batching `expand_dimensions` requires mapped data and \
                               extent inputs for ragged binding `data`",
        ));
    }

    #[test]
    fn test_custom_call_batching_ragged_contract_vectorized_invariant_bindings() {
        // Unchanged replicated inputs cannot carry the full batch extent that a ragged binding describes.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [4]));
        let extent = trace.input(ArrayType::scalar(DataType::I32));
        let mapped = trace.input(ArrayType::new_static(DataType::F32, [3, 2]));
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace, 3);
        let operation = CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [4])])
            .with_batching(CustomCallBatching::Vectorized)
            .with_ragged_contract(preserved_ragged_contract(length));
        assert!(matches!(
            operation.batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayBatch::replicated(packed),
                    ArrayBatch::replicated(extent),
                    ArrayBatch::new(mapped, BatchAxis::new(0)).unwrap(),
                ],
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "custom call `ryft.test.ragged` batching `vectorized` requires mapped data and extent \
                               inputs for ragged binding `data`",
        ));
    }

    #[test]
    fn test_custom_call_differentiation() {
        // A live tangent requires derivative rules that an opaque kernel cannot provide.
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [2]));
        let output = builder.add_instruction(operation.clone(), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert!(matches!(
            program.jvp(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "custom call `ryft.test.add_one` has no differentiation rule; call it through a \
                               `custom_function` with derivative rules to provide one",
        ));

        // Invoke the rule directly so that the driver's zero-tangent shortcut cannot hide a live tangent.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F32, [2]));
        let tangent = context.input(ArrayType::new_static(DataType::F32, [2]));
        assert!(matches!(
            operation.jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new(input, MaybeZero::Value(tangent)).unwrap()],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "custom call `ryft.test.add_one` has no differentiation rule; call it through a \
                               `custom_function` with derivative rules to provide one",
        ));
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_custom_call_differentiation_zero_tangents() {
        // Structural-zero input tangents replay the call on the primals and pair each output with a zero tangent.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F32, [2]));
        let outputs = CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])])
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(input).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].tangent().is_zero());
        assert_eq!(outputs[0].tangent().r#type().as_ref(), &ArrayType::new_static(DataType::F32, [2]));
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].primal().atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = custom_call [target=ryft.test.add_one] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_differentiation_without_inputs() {
        // A zero-input call has no input tangent at all, so its outputs carry structural-zero tangents.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let outputs = CustomCallOperation::new("ryft.test.source", vec![ArrayType::new_static(DataType::F32, [2])])
            .with_side_effect()
            .jvp(&DifferentiationContext::fused(context.clone()), &EmptyRegionDriver, &[])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].tangent().is_zero());
        assert_eq!(outputs[0].tangent().r#type().as_ref(), &ArrayType::new_static(DataType::F32, [2]));
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].primal().atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f32[2] = custom_call [target=ryft.test.source, has_side_effect=true]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_differentiation_array_ir() {
        // The mixed universe follows the same rule as the homogeneous one.
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])]);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [2]).into());
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            program.jvp(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "custom call `ryft.test.add_one` has no differentiation rule; call it through a \
                               `custom_function` with derivative rules to provide one",
        ));

        // A zero-input mixed call replays unchanged with structural-zero output tangents.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let outputs = CustomCallOperation::new("ryft.test.source", vec![ArrayType::new_static(DataType::F32, [2])])
            .jvp_in_parent(&DifferentiationContext::fused(context.clone()), &EmptyRegionDriver, &[])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].tangent().is_zero());
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![outputs[0].primal().atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f32[2] = custom_call [target=ryft.test.source]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])]),
            input_types = [ArrayType::new_static(DataType::F32, [2])],
        );
    }

    #[test]
    fn test_custom_call_transposition_array_ir() {
        // Transposition of the mixed carrier routes through the composite dispatch to the projected nonlinear rule.
        check_operation_transposition!(
            @rejected,
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])]),
            input_types = [ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]))],
        );
    }

    #[test]
    fn test_array_custom_call() {
        // The reference backend has no foreign-kernel registry, with or without inputs.
        let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])]);
        assert!(matches!(
            Array::custom_call(&operation, [&Array::vector(vec![1.0f32, 2.0]).unwrap()]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "the reference array backend cannot execute the foreign kernel `kernel`",
        ));
        assert!(matches!(
            Array::custom_call(&operation, std::iter::empty()),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "the reference array backend cannot execute the foreign kernel `kernel`",
        ));
    }

    #[test]
    fn test_array_ir_value_custom_call() {
        // Concrete composite values call the kernel on their array members.
        let operation = CustomCallOperation::new("kernel", vec![ArrayType::new_static(DataType::F32, [2])]);
        assert!(matches!(
            ArrayIrValue::custom_call(&operation, [&ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap())]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "the reference array backend cannot execute the foreign kernel `kernel`",
        ));
    }

    #[test]
    fn test_tracer_custom_call() {
        // Context-carrying values stage the call through the context of their first input.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let operation =
                    CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::new_static(DataType::F32, [2])]);
                Ok(CustomCall::custom_call(&operation, std::slice::from_ref(&x))?.remove(0))
            },
            ArrayType::new_static(DataType::F32, [2]),
        )
        .unwrap();
        assert_eq!(
            program.to_flat_program().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = custom_call [target=ryft.test.add_one] %0
                in (%1)
            "}
            .trim_end(),
        );

        // Without inputs there is no context to dispatch through.
        let operation = CustomCallOperation::new("ryft.test.source", vec![ArrayType::new_static(DataType::F32, [2])]);
        assert!(matches!(
            <DomainTracer<EagerContext<Array, ArrayOperation<Array>>> as CustomCall>::custom_call(
                &operation,
                std::iter::empty(),
            ),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "the custom-call capability dispatches through its first input's context, so calling \
                               `ryft.test.source` with no inputs requires staging the operation through a program \
                               builder instead",
        ));
    }
}
