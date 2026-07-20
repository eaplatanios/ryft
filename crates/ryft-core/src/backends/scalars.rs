//! Contains the reference scalar backend: concrete [`Scalar`] values, the [`ScalarOperation`] family, and the
//! [`ScalarTracingContext`] used to stage scalar programs.
//!
//! This backend serves programs whose values are rank-0 scalars typed by [`DataType`] alone. It is meant primarily
//! for exercising the Ryft tracing, transformation, and interpretation machinery without involving multidimensional
//! arrays or an optimized array backend such as `ryft-xla`: unit tests, doctests, and gradient checks can stage,
//! transform, and interpret complete scalar programs eagerly. [`Scalar`] carries one payload variant per supported
//! [`DataType`] and implements every value-level capability that [`ScalarOperation`] interpretation requires, and
//! [`ScalarOperation`] is the closed operation family over those scalars.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt::Display;

use approx::AbsDiffEq;
use half::{bf16, f16};
use num_complex::Complex;

use ryft_macros::{Operation, Parameter};

use crate::contexts::EagerContext;
use crate::macros::check_types;
use crate::operations::BooleanLike;
use crate::operations::compare::{Compare, CompareOperation, ComparisonDirection};
use crate::operations::complex::{
    ComplexOperation, Conjugate, ConjugateOperation, Imaginary, ImaginaryOperation, Real, RealOperation,
};
use crate::operations::constants::{
    ConstantOperation, Fill, FillOperation, One, OneLike, OneLikeOperation, OneOperation, Zero, ZeroLike,
    ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::{Select, SelectOperation, WhileOperation, WhilePredicate};
use crate::operations::debugging::PrintOperation;
use crate::operations::differentiation::{StopGradient, StopGradientOperation};
use crate::operations::logical::{And, AndOperation, Not, NotOperation, Or, OrOperation, Xor, XorOperation};
use crate::operations::manipulation::{ConvertElementType, ConvertElementTypeOperation};
use crate::operations::math::{
    Abs, AbsOperation, Add, AddOperation, Atan2, Atan2Operation, Ceil, CeilOperation, Cos, CosOperation, Div,
    DivOperation, Erf, ErfOperation, Exp, ExpOperation, Floor, FloorOperation, Log, LogOperation, Logistic,
    LogisticOperation, Maximum, MaximumOperation, Minimum, MinimumOperation, Mul, MulOperation, Neg, NegOperation, Pow,
    PowOperation, Remainder, RemainderOperation, Round, RoundOperation, Rsqrt, RsqrtOperation, Sign, SignOperation,
    Sin, SinOperation, Sqrt, SqrtOperation, Sub, SubOperation, Tanh, TanhOperation,
};
use crate::operations::tag::{Tag, TagOperation};
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::TracingContext;
use crate::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpOperation, CustomVjpTangentOperation,
};
use crate::tracing_v2::rematerialization::RematerializeOperation;
use crate::types::DataType;

// TODO(eaplatanios): Review this module.

/// Closed [`Scalar`] [`Operation`] type for ordinary staged scalar [`Program`](crate::Program)s.
#[derive(Clone, Debug, Operation)]
#[ryft(dispatch(differentiation, transposition))]
pub enum ScalarOperation<V: Value<Type = DataType>> {
    Zero(ZeroOperation<DataType>),
    ZeroLike(ZeroLikeOperation),
    One(OneOperation<DataType>),
    OneLike(OneLikeOperation),
    Constant(ConstantOperation<V>),
    Fill(FillOperation<DataType, Scalar>),
    Abs(AbsOperation),
    Neg(NegOperation),
    Add(AddOperation),
    Sub(SubOperation),
    Mul(MulOperation),
    Div(DivOperation),
    Sin(SinOperation),
    Cos(CosOperation),
    Atan2(Atan2Operation),
    Exp(ExpOperation),
    Log(LogOperation),
    Sqrt(SqrtOperation),
    Rsqrt(RsqrtOperation),
    Tanh(TanhOperation),
    Logistic(LogisticOperation),
    Erf(ErfOperation),
    Pow(PowOperation),
    Sign(SignOperation),
    Floor(FloorOperation),
    Ceil(CeilOperation),
    Round(RoundOperation),
    Maximum(MaximumOperation),
    Minimum(MinimumOperation),
    Remainder(RemainderOperation),
    Not(NotOperation),
    And(AndOperation),
    Or(OrOperation),
    Xor(XorOperation),
    Complex(ComplexOperation),
    Conjugate(ConjugateOperation),
    Real(RealOperation),
    Imaginary(ImaginaryOperation),
    Compare(CompareOperation),
    Select(SelectOperation),
    While(WhileOperation),
    ConvertElementType(ConvertElementTypeOperation),
    StopGradient(StopGradientOperation),
    Tag(TagOperation),
    Rematerialize(RematerializeOperation),
    Print(PrintOperation),
    CustomJvp(CustomJvpOperation),
    CustomVjp(CustomVjpOperation),
    CustomVjpTangent(CustomVjpTangentOperation<DataType>),
}

/// [`TracingContext`] over the scalar universe, pairing [`DataType`] types and [`Scalar`] staged constants with the
/// [`ScalarOperation`] family.
pub type ScalarTracingContext = TracingContext<Scalar, ScalarOperation<Scalar>>;

/// Scalar [`Value`] whose [`Type`](crate::Type) is a [`DataType`] and which is meant to be used primarily for testing
/// the Ryft infrastructure and machinery with programs that do not involve multidimensional arrays.
///
/// Each variant carries the payload of the corresponding [`DataType`]. The [`Token`](Scalar::Token) variant is the
/// payload-free effect-ordering token value of [`DataType::Token`]. [`Zero`](Scalar::Zero) is the unique value of
/// [`DataType::Zero`], used for trivial differential spaces. Neither supports arithmetic, comparisons, or Boolean
/// conversion, and their `zero_like`/`one_like` are the identity. The `F4*` and `F8*` variants carry their exact
/// encoded bits in a `u8`, preserving signed zeros and NaN payloads. Prefer
/// [`Scalar::from_low_precision_float_bits`] when constructing them dynamically because it validates the four-bit
/// format, and use [`Scalar::low_precision_float_bits`] to recover the encoding.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::backends::scalars::Scalar;
/// # use ryft_core::programs::types::Typed;
/// # use ryft_core::types::DataType;
/// let scalar = Scalar::from(1.5f64);
/// assert_eq!(scalar.r#type().into_owned(), DataType::F64);
/// assert_eq!(scalar, 1.5f64);
/// assert_eq!(scalar + Scalar::from(0.5f64), Scalar::from(2.0f64));
/// ```
#[derive(Copy, Clone, Debug, Parameter)]
pub enum Scalar {
    Token,
    Zero,
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F4E2M1FN(u8),
    F8E3M4(u8),
    F8E4M3(u8),
    F8E4M3FN(u8),
    F8E4M3FNUZ(u8),
    F8E4M3B11FNUZ(u8),
    F8E5M2(u8),
    F8E5M2FNUZ(u8),
    F8E8M0FNU(u8),
    BF16(bf16),
    F16(f16),
    F32(f32),
    F64(f64),
    C64(Complex<f32>),
    C128(Complex<f64>),
}

// `PartialEq` is implemented manually rather than derived because the low-precision floating-point variants must
// compare their *decoded* values rather than their raw bits, so that signed zeros compare equal (`+0 == -0`) and NaN
// payloads compare unequal to themselves, exactly like the wider IEEE variants. Scalars of different data types never
// compare equal.
impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        if let (Some((left_type, left_bits)), Some((right_type, right_bits))) =
            (self.low_precision_float_parts(), other.low_precision_float_parts())
        {
            return left_type == right_type
                && Self::decode_low_precision_float(left_type, left_bits)
                    == Self::decode_low_precision_float(right_type, right_bits);
        }
        match (self, other) {
            (Scalar::Token, Scalar::Token) => true,
            (Scalar::Zero, Scalar::Zero) => true,
            (Scalar::Bool(left), Scalar::Bool(right)) => left == right,
            (Scalar::I8(left), Scalar::I8(right)) => left == right,
            (Scalar::I16(left), Scalar::I16(right)) => left == right,
            (Scalar::I32(left), Scalar::I32(right)) => left == right,
            (Scalar::I64(left), Scalar::I64(right)) => left == right,
            (Scalar::U8(left), Scalar::U8(right)) => left == right,
            (Scalar::U16(left), Scalar::U16(right)) => left == right,
            (Scalar::U32(left), Scalar::U32(right)) => left == right,
            (Scalar::U64(left), Scalar::U64(right)) => left == right,
            (Scalar::BF16(left), Scalar::BF16(right)) => left == right,
            (Scalar::F16(left), Scalar::F16(right)) => left == right,
            (Scalar::F32(left), Scalar::F32(right)) => left == right,
            (Scalar::F64(left), Scalar::F64(right)) => left == right,
            (Scalar::C64(left), Scalar::C64(right)) => left == right,
            (Scalar::C128(left), Scalar::C128(right)) => left == right,
            _ => false,
        }
    }
}

impl Scalar {
    /// Constructs a low-precision floating-point scalar from its raw bit representation.
    pub fn from_low_precision_float_bits(r#type: DataType, bits: u8) -> Result<Self, ProgramError> {
        if r#type == DataType::F4E2M1FN && bits > 0x0f {
            return Err(
                TypeError { message: format!("raw f4e2m1fn value 0x{bits:02x} does not fit in four bits") }.into()
            );
        }
        Ok(match r#type {
            DataType::F4E2M1FN => Scalar::F4E2M1FN(bits),
            DataType::F8E3M4 => Scalar::F8E3M4(bits),
            DataType::F8E4M3 => Scalar::F8E4M3(bits),
            DataType::F8E4M3FN => Scalar::F8E4M3FN(bits),
            DataType::F8E4M3FNUZ => Scalar::F8E4M3FNUZ(bits),
            DataType::F8E4M3B11FNUZ => Scalar::F8E4M3B11FNUZ(bits),
            DataType::F8E5M2 => Scalar::F8E5M2(bits),
            DataType::F8E5M2FNUZ => Scalar::F8E5M2FNUZ(bits),
            DataType::F8E8M0FNU => Scalar::F8E8M0FNU(bits),
            other => {
                return Err(TypeError { message: format!("data type {other} is not a low-precision float") }.into());
            }
        })
    }

    /// Returns the raw bit representation of this low-precision floating-point scalar, or `None` for other scalars.
    pub fn low_precision_float_bits(&self) -> Option<u8> {
        match self {
            Scalar::F4E2M1FN(bits)
            | Scalar::F8E3M4(bits)
            | Scalar::F8E4M3(bits)
            | Scalar::F8E4M3FN(bits)
            | Scalar::F8E4M3FNUZ(bits)
            | Scalar::F8E4M3B11FNUZ(bits)
            | Scalar::F8E5M2(bits)
            | Scalar::F8E5M2FNUZ(bits)
            | Scalar::F8E8M0FNU(bits) => Some(*bits),
            _ => None,
        }
    }

    /// Returns this scalar's low-precision floating-point [`DataType`] together with its raw bit representation,
    /// or `None` for any other variant. This is the branch guard that lets every capability implementation below
    /// handle all low-precision variants uniformly through [`Self::decode_low_precision_float`] and
    /// [`Self::encode_low_precision_float`] before matching on the remaining variants.
    fn low_precision_float_parts(&self) -> Option<(DataType, u8)> {
        Some(match self {
            Scalar::F4E2M1FN(bits) => (DataType::F4E2M1FN, *bits),
            Scalar::F8E3M4(bits) => (DataType::F8E3M4, *bits),
            Scalar::F8E4M3(bits) => (DataType::F8E4M3, *bits),
            Scalar::F8E4M3FN(bits) => (DataType::F8E4M3FN, *bits),
            Scalar::F8E4M3FNUZ(bits) => (DataType::F8E4M3FNUZ, *bits),
            Scalar::F8E4M3B11FNUZ(bits) => (DataType::F8E4M3B11FNUZ, *bits),
            Scalar::F8E5M2(bits) => (DataType::F8E5M2, *bits),
            Scalar::F8E5M2FNUZ(bits) => (DataType::F8E5M2FNUZ, *bits),
            Scalar::F8E8M0FNU(bits) => (DataType::F8E8M0FNU, *bits),
            _ => return None,
        })
    }

    /// Returns an order-preserving `u64` rank for this scalar within its own data type, or `None` for data types
    /// with no total order (complex, token, and structural-zero scalars). Integer ranks are sign-biased two's
    /// complement, and floating-point ranks follow the IEEE 754 total order
    /// (`-NaN < -∞ < … < -0.0 < +0.0 < … < +∞ < NaN`), which is what stable ranking sorts (see
    /// [`SortOperation`](crate::operations::sort::SortOperation)) compare by. Distinct NaN payloads of one sign may
    /// share a rank, which stable sorts resolve by original position.
    pub(crate) fn total_order_rank(&self) -> Option<u64> {
        /// Maps an `f64` to its IEEE 754 total-order rank.
        fn float_rank(value: f64) -> u64 {
            let bits = value.to_bits();
            if bits >> 63 == 1 { !bits } else { bits | (1 << 63) }
        }

        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Some(float_rank(Self::decode_low_precision_float(r#type, bits)));
        }
        Some(match self {
            Scalar::Bool(value) => u64::from(*value),
            Scalar::I8(value) => (*value as u64) ^ (1 << 63),
            Scalar::I16(value) => (*value as u64) ^ (1 << 63),
            Scalar::I32(value) => (*value as u64) ^ (1 << 63),
            Scalar::I64(value) => (*value as u64) ^ (1 << 63),
            Scalar::U8(value) => u64::from(*value),
            Scalar::U16(value) => u64::from(*value),
            Scalar::U32(value) => u64::from(*value),
            Scalar::U64(value) => *value,
            Scalar::BF16(value) => float_rank(value.to_f64()),
            Scalar::F16(value) => float_rank(value.to_f64()),
            Scalar::F32(value) => float_rank(f64::from(*value)),
            Scalar::F64(value) => float_rank(*value),
            _ => return None,
        })
    }

    /// Decodes the raw bit representation of a low-precision floating-point scalar into the exact `f64` value it
    /// denotes, driven by a per-format table of exponent/mantissa widths, biases, NaN encodings, and infinity
    /// support. `f8e8m0fnu` is handled separately because its finite values are bias-127 powers of two with no sign or
    /// mantissa bits and `0xff` is its sole NaN encoding.
    fn decode_low_precision_float(r#type: DataType, bits: u8) -> f64 {
        if r#type == DataType::F8E8M0FNU {
            return if bits == 0xff { f64::NAN } else { 2.0f64.powi(i32::from(bits) - 127) };
        }
        let (total_bits, exponent_bits, mantissa_bits, bias, nan_bits, has_infinity) = match r#type {
            DataType::F4E2M1FN => (4, 2, 1, 1, None, false),
            DataType::F8E3M4 => (8, 3, 4, 3, None, true),
            DataType::F8E4M3 => (8, 4, 3, 7, None, true),
            DataType::F8E4M3FN => (8, 4, 3, 7, Some(0x7f), false),
            DataType::F8E4M3FNUZ => (8, 4, 3, 8, Some(0x80), false),
            DataType::F8E4M3B11FNUZ => (8, 4, 3, 11, Some(0x80), false),
            DataType::F8E5M2 => (8, 5, 2, 15, None, true),
            DataType::F8E5M2FNUZ => (8, 5, 2, 16, Some(0x80), false),
            _ => unreachable!("only low-precision floating-point data types reach this helper"),
        };
        let bits = bits & if total_bits == 8 { u8::MAX } else { (1 << total_bits) - 1 };
        if nan_bits == Some(bits) {
            return f64::NAN;
        }
        let sign = if bits & (1 << (total_bits - 1)) == 0 { 1.0 } else { -1.0 };
        let mantissa_mask = (1 << mantissa_bits) - 1;
        let mantissa = bits & mantissa_mask;
        let exponent_mask = (1 << exponent_bits) - 1;
        let exponent = (bits >> mantissa_bits) & exponent_mask;
        if has_infinity && exponent == exponent_mask {
            return if mantissa == 0 { sign * f64::INFINITY } else { f64::NAN };
        }
        if exponent == 0 {
            sign * 2.0f64.powi(1 - bias) * f64::from(mantissa) / f64::from(1 << mantissa_bits)
        } else {
            sign * 2.0f64.powi(i32::from(exponent) - bias) * (1.0 + f64::from(mantissa) / f64::from(1 << mantissa_bits))
        }
    }

    /// Returns the exactly widened floating-point payload backing this scalar's [`approx::AbsDiffEq`]
    /// implementations, or `None` for a variant that carries no real floating-point payload (Booleans, integers,
    /// and complex values).
    fn floating_point_payload(self) -> Option<f64> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Some(Self::decode_low_precision_float(r#type, bits));
        }
        match self {
            Scalar::BF16(value) => Some(value.to_f64()),
            Scalar::F16(value) => Some(value.to_f64()),
            Scalar::F32(value) => Some(f64::from(value)),
            Scalar::F64(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the exactly widened complex payload backing this scalar's [`approx::AbsDiffEq`] implementations, or
    /// `None` for a non-complex variant.
    fn complex_payload(self) -> Option<Complex<f64>> {
        match self {
            Scalar::C64(value) => Some(Complex::new(f64::from(value.re), f64::from(value.im))),
            Scalar::C128(value) => Some(value),
            _ => None,
        }
    }

    /// Encodes `value` into the nearest representable low-precision floating-point scalar of the provided
    /// [`DataType`], rounding ties toward even bit patterns by exhaustively scanning the format's tiny value space.
    /// NaNs map to the format's canonical NaN encoding (or fail for formats without one), and zeros preserve their
    /// sign where the format distinguishes it.
    fn encode_low_precision_float(r#type: DataType, value: f64) -> Result<Self, ProgramError> {
        let (maximum_bits, canonical_nan) = match r#type {
            DataType::F4E2M1FN => (0x0f, None),
            DataType::F8E3M4 => (0xff, Some(0x71)),
            DataType::F8E4M3 => (0xff, Some(0x79)),
            DataType::F8E4M3FN => (0xff, Some(0x7f)),
            DataType::F8E4M3FNUZ | DataType::F8E4M3B11FNUZ => (0xff, Some(0x80)),
            DataType::F8E5M2 => (0xff, Some(0x7d)),
            DataType::F8E5M2FNUZ => (0xff, Some(0x80)),
            DataType::F8E8M0FNU => (0xff, Some(0xff)),
            // This is a private helper whose callers only ever pass a low-precision floating-point data type
            // (they branch on `low_precision_float_parts` first), so reaching another data type is an internal
            // invariant violation rather than a recoverable error.
            other => unreachable!("{} is not a low-precision floating-point data type", other),
        };
        if value.is_nan() {
            return match canonical_nan {
                Some(bits) => Scalar::from_low_precision_float_bits(r#type, bits),
                None => {
                    Err(TypeError { message: format!("data type {type} cannot represent NaN", type = r#type) }.into())
                }
            };
        }
        if value == 0.0 {
            if r#type == DataType::F8E8M0FNU {
                return Err(TypeError { message: "data type f8e8m0fnu cannot represent zero".to_string() }.into());
            }
            let has_signed_zero =
                !matches!(r#type, DataType::F8E4M3FNUZ | DataType::F8E4M3B11FNUZ | DataType::F8E5M2FNUZ);
            let bits = if has_signed_zero && value.is_sign_negative() {
                if r#type == DataType::F4E2M1FN { 0x08 } else { 0x80 }
            } else {
                0
            };
            return Self::from_low_precision_float_bits(r#type, bits);
        }
        let mut best_bits = None;
        let mut best_distance = f64::INFINITY;
        for bits in 0..=maximum_bits {
            let candidate = Self::decode_low_precision_float(r#type, bits);
            if candidate.is_nan() || candidate.is_infinite() != value.is_infinite() {
                continue;
            }
            let distance = (candidate - value).abs();
            if distance < best_distance || (distance == best_distance && bits & 1 == 0) {
                best_bits = Some(bits);
                best_distance = distance;
            }
        }
        let bits = best_bits.ok_or_else(|| TypeError {
            message: format!("data type {type} cannot represent {value}", type = r#type),
        })?;
        Self::from_low_precision_float_bits(r#type, bits)
    }
}

// `PartialOrd` is implemented manually rather than derived because the complex variants are unordered: same-variant
// comparisons delegate to the payload's own partial order (which does not exist for complex payloads), and
// cross-variant comparisons are `None` (the derived implementation would have ordered them by variant declaration
// order, which is meaningless for scalars of different data types).
impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if let (Some((left_type, left_bits)), Some((right_type, right_bits))) =
            (self.low_precision_float_parts(), other.low_precision_float_parts())
        {
            return (left_type == right_type)
                .then(|| {
                    Self::decode_low_precision_float(left_type, left_bits)
                        .partial_cmp(&Self::decode_low_precision_float(right_type, right_bits))
                })
                .flatten();
        }
        match (self, other) {
            (Scalar::Zero, Scalar::Zero) => Some(Ordering::Equal),
            (Scalar::Bool(left), Scalar::Bool(right)) => left.partial_cmp(right),
            (Scalar::I8(left), Scalar::I8(right)) => left.partial_cmp(right),
            (Scalar::I16(left), Scalar::I16(right)) => left.partial_cmp(right),
            (Scalar::I32(left), Scalar::I32(right)) => left.partial_cmp(right),
            (Scalar::I64(left), Scalar::I64(right)) => left.partial_cmp(right),
            (Scalar::U8(left), Scalar::U8(right)) => left.partial_cmp(right),
            (Scalar::U16(left), Scalar::U16(right)) => left.partial_cmp(right),
            (Scalar::U32(left), Scalar::U32(right)) => left.partial_cmp(right),
            (Scalar::U64(left), Scalar::U64(right)) => left.partial_cmp(right),
            (Scalar::BF16(left), Scalar::BF16(right)) => left.partial_cmp(right),
            (Scalar::F16(left), Scalar::F16(right)) => left.partial_cmp(right),
            (Scalar::F32(left), Scalar::F32(right)) => left.partial_cmp(right),
            (Scalar::F64(left), Scalar::F64(right)) => left.partial_cmp(right),
            _ => None,
        }
    }
}

impl Display for Scalar {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Display::fmt(&Self::decode_low_precision_float(r#type, bits), formatter);
        }
        match self {
            Scalar::Token => formatter.write_str("token"),
            Scalar::Zero => formatter.write_str("zero"),
            Scalar::Bool(value) => Display::fmt(value, formatter),
            Scalar::I8(value) => Display::fmt(value, formatter),
            Scalar::I16(value) => Display::fmt(value, formatter),
            Scalar::I32(value) => Display::fmt(value, formatter),
            Scalar::I64(value) => Display::fmt(value, formatter),
            Scalar::U8(value) => Display::fmt(value, formatter),
            Scalar::U16(value) => Display::fmt(value, formatter),
            Scalar::U32(value) => Display::fmt(value, formatter),
            Scalar::U64(value) => Display::fmt(value, formatter),
            // The low-precision variants were already handled through `low_precision_float_parts` before this
            // match, which must nevertheless remain exhaustive.
            Scalar::F4E2M1FN(_)
            | Scalar::F8E3M4(_)
            | Scalar::F8E4M3(_)
            | Scalar::F8E4M3FN(_)
            | Scalar::F8E4M3FNUZ(_)
            | Scalar::F8E4M3B11FNUZ(_)
            | Scalar::F8E5M2(_)
            | Scalar::F8E5M2FNUZ(_)
            | Scalar::F8E8M0FNU(_) => unreachable!("handled before the match"),
            Scalar::BF16(value) => Display::fmt(value, formatter),
            Scalar::F16(value) => Display::fmt(value, formatter),
            Scalar::F32(value) => Display::fmt(value, formatter),
            Scalar::F64(value) => Display::fmt(value, formatter),
            // Complex scalars render as `<real>+<imaginary>i` (e.g., `1.5+2i`), folding a negative imaginary part's
            // sign into the separator (e.g., `1.5-2i`) so that the rendering reads as ordinary complex notation.
            Scalar::C64(value) if value.im.is_sign_negative() => write!(formatter, "{}-{}i", value.re, -value.im),
            Scalar::C64(value) => write!(formatter, "{}+{}i", value.re, value.im),
            Scalar::C128(value) if value.im.is_sign_negative() => write!(formatter, "{}-{}i", value.re, -value.im),
            Scalar::C128(value) => write!(formatter, "{}+{}i", value.re, value.im),
        }
    }
}

impl Typed for Scalar {
    type Type = DataType;

    fn r#type(&self) -> Cow<'_, DataType> {
        Cow::Owned(match self {
            Scalar::Token => DataType::Token,
            Scalar::Zero => DataType::Zero,
            Scalar::Bool(_) => DataType::Boolean,
            Scalar::I8(_) => DataType::I8,
            Scalar::I16(_) => DataType::I16,
            Scalar::I32(_) => DataType::I32,
            Scalar::I64(_) => DataType::I64,
            Scalar::U8(_) => DataType::U8,
            Scalar::U16(_) => DataType::U16,
            Scalar::U32(_) => DataType::U32,
            Scalar::U64(_) => DataType::U64,
            Scalar::F4E2M1FN(_) => DataType::F4E2M1FN,
            Scalar::F8E3M4(_) => DataType::F8E3M4,
            Scalar::F8E4M3(_) => DataType::F8E4M3,
            Scalar::F8E4M3FN(_) => DataType::F8E4M3FN,
            Scalar::F8E4M3FNUZ(_) => DataType::F8E4M3FNUZ,
            Scalar::F8E4M3B11FNUZ(_) => DataType::F8E4M3B11FNUZ,
            Scalar::F8E5M2(_) => DataType::F8E5M2,
            Scalar::F8E5M2FNUZ(_) => DataType::F8E5M2FNUZ,
            Scalar::F8E8M0FNU(_) => DataType::F8E8M0FNU,
            Scalar::BF16(_) => DataType::BF16,
            Scalar::F16(_) => DataType::F16,
            Scalar::F32(_) => DataType::F32,
            Scalar::F64(_) => DataType::F64,
            Scalar::C64(_) => DataType::C64,
            Scalar::C128(_) => DataType::C128,
        })
    }
}

impl Value for Scalar {
    type DispatchDomain = EagerContext<Scalar>;
    type ExecutionDomain = EagerContext<Scalar, ScalarOperation<Scalar>>;

    #[inline]
    fn dispatch_domain(&self) -> EagerContext<Scalar> {
        EagerContext::new()
    }

    #[inline]
    fn execution_domain(&self) -> EagerContext<Scalar, ScalarOperation<Scalar>> {
        EagerContext::new()
    }
}

/// Implements the conversion from a supported Rust primitive into the corresponding [`Scalar`] variant. These let
/// later stages and numeric-literal tests write `Scalar::from(0.0)` without naming the variant explicitly.
macro_rules! impl_from_primitive_for_scalar {
    ($ty:ty, $variant:ident) => {
        impl From<$ty> for Scalar {
            fn from(value: $ty) -> Self {
                Scalar::$variant(value)
            }
        }
    };
}

impl_from_primitive_for_scalar!(bool, Bool);
impl_from_primitive_for_scalar!(i8, I8);
impl_from_primitive_for_scalar!(i16, I16);
impl_from_primitive_for_scalar!(i32, I32);
impl_from_primitive_for_scalar!(i64, I64);
impl_from_primitive_for_scalar!(u8, U8);
impl_from_primitive_for_scalar!(u16, U16);
impl_from_primitive_for_scalar!(u32, U32);
impl_from_primitive_for_scalar!(u64, U64);
impl_from_primitive_for_scalar!(bf16, BF16);
impl_from_primitive_for_scalar!(f16, F16);
impl_from_primitive_for_scalar!(f32, F32);
impl_from_primitive_for_scalar!(f64, F64);
impl_from_primitive_for_scalar!(Complex<f32>, C64);
impl_from_primitive_for_scalar!(Complex<f64>, C128);

/// Implements equality against a supported Rust primitive, comparing only within the matching variant so that a
/// [`Scalar`] of a different [`DataType`] never compares equal to a primitive (e.g., a `Scalar::F32` is never equal
/// to an `f64`). These let later stages and numeric-literal tests write `scalar == 0.0` directly.
macro_rules! impl_partial_eq_primitive_for_scalar {
    ($ty:ty, $variant:ident) => {
        impl PartialEq<$ty> for Scalar {
            fn eq(&self, other: &$ty) -> bool {
                matches!(self, Scalar::$variant(value) if value == other)
            }
        }

        impl PartialEq<Scalar> for $ty {
            fn eq(&self, other: &Scalar) -> bool {
                matches!(other, Scalar::$variant(value) if value == self)
            }
        }
    };
}

impl_partial_eq_primitive_for_scalar!(bool, Bool);
impl_partial_eq_primitive_for_scalar!(i8, I8);
impl_partial_eq_primitive_for_scalar!(i16, I16);
impl_partial_eq_primitive_for_scalar!(i32, I32);
impl_partial_eq_primitive_for_scalar!(i64, I64);
impl_partial_eq_primitive_for_scalar!(u8, U8);
impl_partial_eq_primitive_for_scalar!(u16, U16);
impl_partial_eq_primitive_for_scalar!(u32, U32);
impl_partial_eq_primitive_for_scalar!(u64, U64);
impl_partial_eq_primitive_for_scalar!(bf16, BF16);
impl_partial_eq_primitive_for_scalar!(f16, F16);
impl_partial_eq_primitive_for_scalar!(f32, F32);
impl_partial_eq_primitive_for_scalar!(f64, F64);
impl_partial_eq_primitive_for_scalar!(Complex<f32>, C64);
impl_partial_eq_primitive_for_scalar!(Complex<f64>, C128);

/// Approximate equality against a bare `f64`, serving the [`approx`] assertion macros in tests (e.g.,
/// `assert_abs_diff_eq!(gradient, expected, epsilon = 1e-9)` where `gradient` is a [`Scalar`]). A floating-point
/// variant compares its exactly widened payload within `epsilon`, while a variant with no real floating-point
/// payload (a Boolean, integer, or complex value) is never approximately equal to a bare `f64`.
impl AbsDiffEq<f64> for Scalar {
    type Epsilon = f64;

    fn default_epsilon() -> f64 {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &f64, epsilon: f64) -> bool {
        match self.floating_point_payload() {
            Some(value) => (value - other).abs() <= epsilon,
            None => false,
        }
    }
}

/// Approximate equality between two [`Scalar`]s, serving the [`approx`] assertion macros in tests. Two
/// floating-point variants compare their exactly widened payloads within `epsilon` (also across variants, e.g., a
/// [`Scalar::F32`] against a [`Scalar::F64`]), two complex variants compare their exactly widened payloads
/// componentwise within `epsilon` (both the real and the imaginary parts must be close), and any other pairing falls
/// back to exact [`PartialEq`] equality, which is the only equality Booleans and integers define.
impl AbsDiffEq for Scalar {
    type Epsilon = f64;

    fn default_epsilon() -> f64 {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: f64) -> bool {
        match (self.floating_point_payload(), other.floating_point_payload()) {
            (Some(left), Some(right)) => (left - right).abs() <= epsilon,
            _ => match (self.complex_payload(), other.complex_payload()) {
                (Some(left), Some(right)) => {
                    (left.re - right.re).abs() <= epsilon && (left.im - right.im).abs() <= epsilon
                }
                _ => self == other,
            },
        }
    }
}

impl<O: Operation<DataType>> Zero<Scalar> for EagerContext<Scalar, O> {
    #[inline]
    fn zero(&self, r#type: &DataType) -> Result<Scalar, ProgramError> {
        Ok(match r#type {
            DataType::Zero => Scalar::Zero,
            DataType::Boolean => Scalar::Bool(false),
            DataType::I8 => Scalar::I8(0),
            DataType::I16 => Scalar::I16(0),
            DataType::I32 => Scalar::I32(0),
            DataType::I64 => Scalar::I64(0),
            DataType::U8 => Scalar::U8(0),
            DataType::U16 => Scalar::U16(0),
            DataType::U32 => Scalar::U32(0),
            DataType::U64 => Scalar::U64(0),
            DataType::F4E2M1FN => Scalar::F4E2M1FN(0),
            DataType::F8E3M4 => Scalar::F8E3M4(0),
            DataType::F8E4M3 => Scalar::F8E4M3(0),
            DataType::F8E4M3FN => Scalar::F8E4M3FN(0),
            DataType::F8E4M3FNUZ => Scalar::F8E4M3FNUZ(0),
            DataType::F8E4M3B11FNUZ => Scalar::F8E4M3B11FNUZ(0),
            DataType::F8E5M2 => Scalar::F8E5M2(0),
            DataType::F8E5M2FNUZ => Scalar::F8E5M2FNUZ(0),
            DataType::BF16 => Scalar::BF16(bf16::ZERO),
            DataType::F16 => Scalar::F16(f16::ZERO),
            DataType::F32 => Scalar::F32(0.0),
            DataType::F64 => Scalar::F64(0.0),
            DataType::C64 => Scalar::C64(Complex::new(0.0, 0.0)),
            DataType::C128 => Scalar::C128(Complex::new(0.0, 0.0)),
            other => {
                return Err(TypeError { message: format!("data type {other} cannot represent zero") }.into());
            }
        })
    }
}

impl ZeroLike for Scalar {
    #[inline]
    fn zero_like(&self) -> Self {
        match self {
            Scalar::Token => Scalar::Token,
            Scalar::Zero => Scalar::Zero,
            Scalar::Bool(_) => Scalar::Bool(false),
            Scalar::I8(_) => Scalar::I8(0),
            Scalar::I16(_) => Scalar::I16(0),
            Scalar::I32(_) => Scalar::I32(0),
            Scalar::I64(_) => Scalar::I64(0),
            Scalar::U8(_) => Scalar::U8(0),
            Scalar::U16(_) => Scalar::U16(0),
            Scalar::U32(_) => Scalar::U32(0),
            Scalar::U64(_) => Scalar::U64(0),
            Scalar::F4E2M1FN(_) => Scalar::F4E2M1FN(0),
            Scalar::F8E3M4(_) => Scalar::F8E3M4(0),
            Scalar::F8E4M3(_) => Scalar::F8E4M3(0),
            Scalar::F8E4M3FN(_) => Scalar::F8E4M3FN(0),
            Scalar::F8E4M3FNUZ(_) => Scalar::F8E4M3FNUZ(0),
            Scalar::F8E4M3B11FNUZ(_) => Scalar::F8E4M3B11FNUZ(0),
            Scalar::F8E5M2(_) => Scalar::F8E5M2(0),
            Scalar::F8E5M2FNUZ(_) => Scalar::F8E5M2FNUZ(0),
            Scalar::F8E8M0FNU(_) => *self,
            Scalar::BF16(_) => Scalar::BF16(bf16::ZERO),
            Scalar::F16(_) => Scalar::F16(f16::ZERO),
            Scalar::F32(_) => Scalar::F32(0.0),
            Scalar::F64(_) => Scalar::F64(0.0),
            Scalar::C64(_) => Scalar::C64(Complex::new(0.0, 0.0)),
            Scalar::C128(_) => Scalar::C128(Complex::new(0.0, 0.0)),
        }
    }
}

impl<O: Operation<DataType>> One<Scalar> for EagerContext<Scalar, O> {
    #[inline]
    fn one(&self, r#type: &DataType) -> Result<Scalar, ProgramError> {
        Ok(match r#type {
            DataType::Boolean => Scalar::Bool(true),
            DataType::I8 => Scalar::I8(1),
            DataType::I16 => Scalar::I16(1),
            DataType::I32 => Scalar::I32(1),
            DataType::I64 => Scalar::I64(1),
            DataType::U8 => Scalar::U8(1),
            DataType::U16 => Scalar::U16(1),
            DataType::U32 => Scalar::U32(1),
            DataType::U64 => Scalar::U64(1),
            DataType::F4E2M1FN => Scalar::F4E2M1FN(0x02),
            DataType::F8E3M4 => Scalar::F8E3M4(0x30),
            DataType::F8E4M3 => Scalar::F8E4M3(0x38),
            DataType::F8E4M3FN => Scalar::F8E4M3FN(0x38),
            DataType::F8E4M3FNUZ => Scalar::F8E4M3FNUZ(0x40),
            DataType::F8E4M3B11FNUZ => Scalar::F8E4M3B11FNUZ(0x58),
            DataType::F8E5M2 => Scalar::F8E5M2(0x3c),
            DataType::F8E5M2FNUZ => Scalar::F8E5M2FNUZ(0x40),
            DataType::F8E8M0FNU => Scalar::F8E8M0FNU(0x7f),
            DataType::BF16 => Scalar::BF16(bf16::ONE),
            DataType::F16 => Scalar::F16(f16::ONE),
            DataType::F32 => Scalar::F32(1.0),
            DataType::F64 => Scalar::F64(1.0),
            DataType::C64 => Scalar::C64(Complex::new(1.0, 0.0)),
            DataType::C128 => Scalar::C128(Complex::new(1.0, 0.0)),
            other => {
                return Err(TypeError { message: format!("data type {other} cannot represent one") }.into());
            }
        })
    }
}

impl<O: Operation<DataType>> Fill<Scalar, Scalar> for EagerContext<Scalar, O> {
    #[inline]
    fn fill(&self, r#type: &DataType, value: Scalar) -> Result<Scalar, ProgramError> {
        // Mirroring the array backend's fill, the fill element must be losslessly representable in the target data
        // type, so this uses the promotion-checked conversion rather than an unchecked cast: filling a real scalar
        // slot with a complex value (or any other narrowing fill) is rejected instead of silently discarding payload.
        value.promote_element_type(*r#type)
    }
}

impl OneLike for Scalar {
    #[inline]
    fn one_like(&self) -> Self {
        match self {
            Scalar::Token => Scalar::Token,
            Scalar::Zero => Scalar::Zero,
            Scalar::Bool(_) => Scalar::Bool(true),
            Scalar::I8(_) => Scalar::I8(1),
            Scalar::I16(_) => Scalar::I16(1),
            Scalar::I32(_) => Scalar::I32(1),
            Scalar::I64(_) => Scalar::I64(1),
            Scalar::U8(_) => Scalar::U8(1),
            Scalar::U16(_) => Scalar::U16(1),
            Scalar::U32(_) => Scalar::U32(1),
            Scalar::U64(_) => Scalar::U64(1),
            Scalar::F4E2M1FN(_) => Scalar::F4E2M1FN(0x02),
            Scalar::F8E3M4(_) => Scalar::F8E3M4(0x30),
            Scalar::F8E4M3(_) => Scalar::F8E4M3(0x38),
            Scalar::F8E4M3FN(_) => Scalar::F8E4M3FN(0x38),
            Scalar::F8E4M3FNUZ(_) => Scalar::F8E4M3FNUZ(0x40),
            Scalar::F8E4M3B11FNUZ(_) => Scalar::F8E4M3B11FNUZ(0x58),
            Scalar::F8E5M2(_) => Scalar::F8E5M2(0x3c),
            Scalar::F8E5M2FNUZ(_) => Scalar::F8E5M2FNUZ(0x40),
            Scalar::F8E8M0FNU(_) => Scalar::F8E8M0FNU(0x7f),
            Scalar::BF16(_) => Scalar::BF16(bf16::ONE),
            Scalar::F16(_) => Scalar::F16(f16::ONE),
            Scalar::F32(_) => Scalar::F32(1.0),
            Scalar::F64(_) => Scalar::F64(1.0),
            Scalar::C64(_) => Scalar::C64(Complex::new(1.0, 0.0)),
            Scalar::C128(_) => Scalar::C128(Complex::new(1.0, 0.0)),
        }
    }
}

impl Abs for Scalar {
    /// Computes the elementwise absolute value of this [`Scalar`]: the magnitude `|z|` (with a real result) for the
    /// complex variants, and `|x|` for the signed-integer and floating-point variants. Any other variant returns a
    /// [`TypeError`]. The minimum value of a signed-integer variant has no positive counterpart in the same type and
    /// therefore wraps to itself, matching two's-complement integer absolute-value semantics without panicking.
    fn abs(&self) -> Result<Self, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Self::encode_low_precision_float(r#type, Self::decode_low_precision_float(r#type, bits).abs());
        }
        Ok(match self {
            Scalar::I8(value) => Scalar::I8(value.wrapping_abs()),
            Scalar::I16(value) => Scalar::I16(value.wrapping_abs()),
            Scalar::I32(value) => Scalar::I32(value.wrapping_abs()),
            Scalar::I64(value) => Scalar::I64(value.wrapping_abs()),
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().abs())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().abs())),
            Scalar::F32(value) => Scalar::F32(value.abs()),
            Scalar::F64(value) => Scalar::F64(value.abs()),
            Scalar::C64(value) => Scalar::F32(value.norm()),
            Scalar::C128(value) => Scalar::F64(value.norm()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the absolute value of a scalar of data type {}", other.r#type(),),
                }
                .into());
            }
        })
    }
}

impl Neg for Scalar {
    #[inline]
    fn neg(&self) -> Result<Scalar, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            if r#type == DataType::F8E8M0FNU {
                return Err(TypeError { message: "cannot negate a scalar of data type f8e8m0fnu".to_string() }.into());
            }
            return Self::encode_low_precision_float(r#type, -Self::decode_low_precision_float(r#type, bits));
        }
        Ok(match *self {
            Scalar::I8(value) => Scalar::I8(value.wrapping_neg()),
            Scalar::I16(value) => Scalar::I16(value.wrapping_neg()),
            Scalar::I32(value) => Scalar::I32(value.wrapping_neg()),
            Scalar::I64(value) => Scalar::I64(value.wrapping_neg()),
            Scalar::U8(value) => Scalar::U8(value.wrapping_neg()),
            Scalar::U16(value) => Scalar::U16(value.wrapping_neg()),
            Scalar::U32(value) => Scalar::U32(value.wrapping_neg()),
            Scalar::U64(value) => Scalar::U64(value.wrapping_neg()),
            // The low-precision variants were already handled through `low_precision_float_parts` before this
            // match, which must nevertheless remain exhaustive.
            Scalar::F4E2M1FN(_)
            | Scalar::F8E3M4(_)
            | Scalar::F8E4M3(_)
            | Scalar::F8E4M3FN(_)
            | Scalar::F8E4M3FNUZ(_)
            | Scalar::F8E4M3B11FNUZ(_)
            | Scalar::F8E5M2(_)
            | Scalar::F8E5M2FNUZ(_)
            | Scalar::F8E8M0FNU(_) => unreachable!("handled before the match"),
            Scalar::BF16(value) => Scalar::BF16(-value),
            Scalar::F16(value) => Scalar::F16(-value),
            Scalar::F32(value) => Scalar::F32(-value),
            Scalar::F64(value) => Scalar::F64(-value),
            Scalar::C64(value) => Scalar::C64(-value),
            Scalar::C128(value) => Scalar::C128(-value),
            other => {
                return Err(
                    TypeError { message: format!("cannot negate a scalar of data type {}", other.r#type()) }.into()
                );
            }
        })
    }
}

impl std::ops::Neg for Scalar {
    type Output = Scalar;

    #[inline]
    fn neg(self) -> Scalar {
        Neg::neg(&self).unwrap_or_else(|error| panic!("{error}"))
    }
}

/// Promotes two numeric scalars to the common data type used by a binary arithmetic operation.
fn promote_scalar_arithmetic_operands(
    left: &Scalar,
    right: &Scalar,
    operation_name: &str,
) -> Result<(Scalar, Scalar), ProgramError> {
    let input_types = [left.r#type().into_owned(), right.r#type().into_owned()];
    check_types!(@numeric, operation_name, input_types);
    let target = DataType::promoted(&input_types).map_err(|error| TypeError { message: error.to_string() })?;
    Ok((left.convert_element_type(target)?, right.convert_element_type(target)?))
}

/// Implements a fallible binary arithmetic capability (e.g., [`Add`]) together with its panicking [`std::ops`]
/// counterpart for [`Scalar`]. Numeric operands are first promoted to their common [`DataType`]. Integer operations
/// use deterministic two's-complement wrapping semantics, while low-precision floating-point operations compute
/// through decoded `f64` values and re-encode the result.
macro_rules! impl_binary_arithmetic_for_scalar {
    ($trait:ident, $method:ident, $integer_method:ident, $operator:tt) => {
        impl $trait for Scalar {
            #[inline]
            fn $method(&self, rhs: &Scalar) -> Result<Scalar, ProgramError> {
                let (left, right) = promote_scalar_arithmetic_operands(self, rhs, stringify!($method))?;
                if let (Some((left_type, left_bits)), Some((right_type, right_bits))) =
                    (left.low_precision_float_parts(), right.low_precision_float_parts())
                {
                    debug_assert_eq!(left_type, right_type);
                    return Scalar::encode_low_precision_float(
                        left_type,
                        Scalar::decode_low_precision_float(left_type, left_bits)
                            $operator Scalar::decode_low_precision_float(right_type, right_bits),
                    );
                }
                Ok(match (left, right) {
                    (Scalar::I8(left), Scalar::I8(right)) => Scalar::I8(left.$integer_method(right)),
                    (Scalar::I16(left), Scalar::I16(right)) => Scalar::I16(left.$integer_method(right)),
                    (Scalar::I32(left), Scalar::I32(right)) => Scalar::I32(left.$integer_method(right)),
                    (Scalar::I64(left), Scalar::I64(right)) => Scalar::I64(left.$integer_method(right)),
                    (Scalar::U8(left), Scalar::U8(right)) => Scalar::U8(left.$integer_method(right)),
                    (Scalar::U16(left), Scalar::U16(right)) => Scalar::U16(left.$integer_method(right)),
                    (Scalar::U32(left), Scalar::U32(right)) => Scalar::U32(left.$integer_method(right)),
                    (Scalar::U64(left), Scalar::U64(right)) => Scalar::U64(left.$integer_method(right)),
                    (Scalar::BF16(left), Scalar::BF16(right)) => Scalar::BF16(left $operator right),
                    (Scalar::F16(left), Scalar::F16(right)) => Scalar::F16(left $operator right),
                    (Scalar::F32(left), Scalar::F32(right)) => Scalar::F32(left $operator right),
                    (Scalar::F64(left), Scalar::F64(right)) => Scalar::F64(left $operator right),
                    (Scalar::C64(left), Scalar::C64(right)) => Scalar::C64(left $operator right),
                    (Scalar::C128(left), Scalar::C128(right)) => Scalar::C128(left $operator right),
                    (left, right) => {
                        return Err(TypeError {
                            message: format!(
                                "cannot apply `{}` to scalars of data types {} and {}",
                                stringify!($method),
                                left.r#type(),
                                right.r#type(),
                            ),
                        }
                        .into());
                    }
                })
            }
        }

        impl std::ops::$trait for Scalar {
            type Output = Scalar;

            #[inline]
            fn $method(self, rhs: Scalar) -> Scalar {
                $trait::$method(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
            }
        }
    };
}

impl_binary_arithmetic_for_scalar!(Add, add, wrapping_add, +);
impl_binary_arithmetic_for_scalar!(Sub, sub, wrapping_sub, -);
impl_binary_arithmetic_for_scalar!(Mul, mul, wrapping_mul, *);

/// Divides two finite complex scalars after scaling both operands by the denominator's largest component. The common
/// scaling leaves the quotient unchanged while preventing avoidable overflow in the direct norm-squared formula.
macro_rules! divide_complex_scalars {
    ($left:expr, $right:expr) => {{
        let left = $left;
        let right = $right;
        let direct = left / right;
        // Preserve `num_complex`'s ordinary rounding and non-finite-value behavior. Recompute only finite inputs whose
        // direct norm-squared formula overflowed to a non-finite result despite a nonzero denominator.
        if direct.re.is_finite() && direct.im.is_finite()
            || !left.re.is_finite()
            || !left.im.is_finite()
            || !right.re.is_finite()
            || !right.im.is_finite()
            || right.re == 0.0 && right.im == 0.0
        {
            direct
        } else if right.im == 0.0 {
            Complex::new(left.re / right.re, left.im / right.re)
        } else if right.re == 0.0 {
            Complex::new(left.im / right.im, -left.re / right.im)
        } else {
            // Normalize by the denominator rather than by all four components. Including the numerator in this scale
            // can round a much smaller, nonzero denominator to zero even though the mathematically correct quotient is
            // merely large or infinite.
            let scale = right.re.abs().max(right.im.abs());
            let left = Complex::new(left.re / scale, left.im / scale);
            let right = Complex::new(right.re / scale, right.im / scale);
            if right.re.abs() >= right.im.abs() {
                let ratio = right.im / right.re;
                let denominator = right.re + right.im * ratio;
                Complex::new((left.re + left.im * ratio) / denominator, (left.im - left.re * ratio) / denominator)
            } else {
                let ratio = right.re / right.im;
                let denominator = right.im + right.re * ratio;
                Complex::new((left.re * ratio + left.im) / denominator, (left.im * ratio - left.re) / denominator)
            }
        }
    }};
}

impl Div for Scalar {
    fn div(&self, rhs: &Scalar) -> Result<Scalar, ProgramError> {
        let (left, right) = promote_scalar_arithmetic_operands(self, rhs, "div")?;
        if let (Some((left_type, left_bits)), Some((right_type, right_bits))) =
            (left.low_precision_float_parts(), right.low_precision_float_parts())
        {
            debug_assert_eq!(left_type, right_type);
            return Scalar::encode_low_precision_float(
                left_type,
                Scalar::decode_low_precision_float(left_type, left_bits)
                    / Scalar::decode_low_precision_float(right_type, right_bits),
            );
        }

        macro_rules! divide_signed_integer {
            ($left:expr, $right:expr, $variant:ident, $data_type:ident, $primitive:ty) => {{
                if $right == 0 {
                    return Err(TypeError {
                        message: format!(
                            "cannot divide an integer scalar of data type {} by zero",
                            DataType::$data_type
                        ),
                    }
                    .into());
                }
                if $left == <$primitive>::MIN && $right == -1 {
                    return Err(TypeError {
                        message: format!(
                            "cannot divide the minimum integer scalar of data type {} by -1",
                            DataType::$data_type,
                        ),
                    }
                    .into());
                }
                Scalar::$variant($left / $right)
            }};
        }

        macro_rules! divide_unsigned_integer {
            ($left:expr, $right:expr, $variant:ident, $data_type:ident) => {{
                if $right == 0 {
                    return Err(TypeError {
                        message: format!(
                            "cannot divide an integer scalar of data type {} by zero",
                            DataType::$data_type
                        ),
                    }
                    .into());
                }
                Scalar::$variant($left / $right)
            }};
        }

        Ok(match (left, right) {
            (Scalar::I8(left), Scalar::I8(right)) => divide_signed_integer!(left, right, I8, I8, i8),
            (Scalar::I16(left), Scalar::I16(right)) => divide_signed_integer!(left, right, I16, I16, i16),
            (Scalar::I32(left), Scalar::I32(right)) => divide_signed_integer!(left, right, I32, I32, i32),
            (Scalar::I64(left), Scalar::I64(right)) => divide_signed_integer!(left, right, I64, I64, i64),
            (Scalar::U8(left), Scalar::U8(right)) => divide_unsigned_integer!(left, right, U8, U8),
            (Scalar::U16(left), Scalar::U16(right)) => divide_unsigned_integer!(left, right, U16, U16),
            (Scalar::U32(left), Scalar::U32(right)) => divide_unsigned_integer!(left, right, U32, U32),
            (Scalar::U64(left), Scalar::U64(right)) => divide_unsigned_integer!(left, right, U64, U64),
            (Scalar::BF16(left), Scalar::BF16(right)) => Scalar::BF16(left / right),
            (Scalar::F16(left), Scalar::F16(right)) => Scalar::F16(left / right),
            (Scalar::F32(left), Scalar::F32(right)) => Scalar::F32(left / right),
            (Scalar::F64(left), Scalar::F64(right)) => Scalar::F64(left / right),
            (Scalar::C64(left), Scalar::C64(right)) => Scalar::C64(divide_complex_scalars!(left, right)),
            (Scalar::C128(left), Scalar::C128(right)) => Scalar::C128(divide_complex_scalars!(left, right)),
            (left, right) => {
                return Err(TypeError {
                    message: format!(
                        "cannot apply `div` to scalars of data types {} and {}",
                        left.r#type(),
                        right.r#type(),
                    ),
                }
                .into());
            }
        })
    }
}

impl std::ops::Div for Scalar {
    type Output = Scalar;

    #[inline]
    fn div(self, rhs: Scalar) -> Scalar {
        Div::div(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl Sin for Scalar {
    /// Computes the elementwise sine of this [`Scalar`]. Only the floating-point and complex variants support sine
    /// (the complex sine being the analytic continuation `sin(z)`); any other variant returns a [`TypeError`].
    fn sin(&self) -> Result<Self, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Self::encode_low_precision_float(r#type, Self::decode_low_precision_float(r#type, bits).sin());
        }
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().sin())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().sin())),
            Scalar::F32(value) => Scalar::F32(value.sin()),
            Scalar::F64(value) => Scalar::F64(value.sin()),
            Scalar::C64(value) => {
                let expm1_imaginary = value.im.exp_m1();
                let expm1_negative_imaginary = (-value.im).exp_m1();
                let sinh_imaginary = (expm1_imaginary - expm1_negative_imaginary) / 2.0;
                let cosh_imaginary = (expm1_imaginary + expm1_negative_imaginary + 2.0) / 2.0;
                let imaginary = value.re.cos() * sinh_imaginary;
                Scalar::C64(Complex::new(
                    if value.re == 0.0 { 0.0 } else { value.re.sin() * cosh_imaginary },
                    imaginary,
                ))
            }
            Scalar::C128(value) => {
                let expm1_imaginary = value.im.exp_m1();
                let expm1_negative_imaginary = (-value.im).exp_m1();
                let sinh_imaginary = (expm1_imaginary - expm1_negative_imaginary) / 2.0;
                let cosh_imaginary = (expm1_imaginary + expm1_negative_imaginary + 2.0) / 2.0;
                let imaginary = value.re.cos() * sinh_imaginary;
                Scalar::C128(Complex::new(
                    if value.re == 0.0 { 0.0 } else { value.re.sin() * cosh_imaginary },
                    imaginary,
                ))
            }
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the sine of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Cos for Scalar {
    /// Computes the elementwise cosine of this [`Scalar`]. Only the floating-point and complex variants support cosine
    /// (the complex cosine being the analytic continuation `cos(z)`); any other variant returns a [`TypeError`].
    fn cos(&self) -> Result<Self, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Self::encode_low_precision_float(r#type, Self::decode_low_precision_float(r#type, bits).cos());
        }
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().cos())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().cos())),
            Scalar::F32(value) => Scalar::F32(value.cos()),
            Scalar::F64(value) => Scalar::F64(value.cos()),
            Scalar::C64(value) => {
                let expm1_imaginary = value.im.exp_m1();
                let expm1_negative_imaginary = (-value.im).exp_m1();
                let sinh_imaginary = (expm1_imaginary - expm1_negative_imaginary) / 2.0;
                let cosh_imaginary = (expm1_imaginary + expm1_negative_imaginary + 2.0) / 2.0;
                let real = value.re.cos() * cosh_imaginary;
                Scalar::C64(Complex::new(real, if value.re == 0.0 { 0.0 } else { -value.re.sin() * sinh_imaginary }))
            }
            Scalar::C128(value) => {
                let expm1_imaginary = value.im.exp_m1();
                let expm1_negative_imaginary = (-value.im).exp_m1();
                let sinh_imaginary = (expm1_imaginary - expm1_negative_imaginary) / 2.0;
                let cosh_imaginary = (expm1_imaginary + expm1_negative_imaginary + 2.0) / 2.0;
                let real = value.re.cos() * cosh_imaginary;
                Scalar::C128(Complex::new(real, if value.re == 0.0 { 0.0 } else { -value.re.sin() * sinh_imaginary }))
            }
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the cosine of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Atan2 for Scalar {
    /// Computes the elementwise two-argument arc tangent `atan2(self, x)` for this [`Scalar`]. Floating-point and
    /// complex operands are promoted to a common data type; any other combination returns a [`TypeError`]. Complex
    /// results use the principal value `-i · log((x + i · self) / sqrt(x² + self²))`.
    fn atan2(&self, x: &Self) -> Result<Self, ProgramError> {
        check_types!(@float, "atan2", [self.r#type().into_owned(), x.r#type().into_owned()]);
        let (y, x) = promote_scalar_arithmetic_operands(self, x, "atan2")?;
        if let (Some((left_type, left_bits)), Some((right_type, right_bits))) =
            (y.low_precision_float_parts(), x.low_precision_float_parts())
        {
            return Self::encode_low_precision_float(
                left_type,
                Self::decode_low_precision_float(left_type, left_bits)
                    .atan2(Self::decode_low_precision_float(right_type, right_bits)),
            );
        }
        Ok(match (y, x) {
            (Scalar::BF16(y), Scalar::BF16(x)) => Scalar::BF16(bf16::from_f32(y.to_f32().atan2(x.to_f32()))),
            (Scalar::F16(y), Scalar::F16(x)) => Scalar::F16(f16::from_f32(y.to_f32().atan2(x.to_f32()))),
            (Scalar::F32(y), Scalar::F32(x)) => Scalar::F32(y.atan2(x)),
            (Scalar::F64(y), Scalar::F64(x)) => Scalar::F64(y.atan2(x)),
            (Scalar::C64(y), Scalar::C64(x)) => {
                let imaginary_unit = Complex::new(0.0, 1.0);
                let radius = (x * x + y * y).sqrt();
                Scalar::C64(-imaginary_unit * divide_complex_scalars!(x + imaginary_unit * y, radius).ln())
            }
            (Scalar::C128(y), Scalar::C128(x)) => {
                let imaginary_unit = Complex::new(0.0, 1.0);
                let radius = (x * x + y * y).sqrt();
                Scalar::C128(-imaginary_unit * divide_complex_scalars!(x + imaginary_unit * y, radius).ln())
            }
            (y, x) => {
                return Err(TypeError {
                    message: format!(
                        "cannot compute the arc tangent of scalars of data types {} and {}",
                        y.r#type(),
                        x.r#type(),
                    ),
                }
                .into());
            }
        })
    }
}

impl Exp for Scalar {
    /// Computes the elementwise natural exponential of this [`Scalar`]. Only the floating-point and complex variants
    /// support the exponential (the complex exponential being the analytic continuation `e^z`); any other variant
    /// returns a [`TypeError`].
    fn exp(&self) -> Result<Self, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Self::encode_low_precision_float(r#type, Self::decode_low_precision_float(r#type, bits).exp());
        }
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().exp())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().exp())),
            Scalar::F32(value) => Scalar::F32(value.exp()),
            Scalar::F64(value) => Scalar::F64(value.exp()),
            Scalar::C64(value) => Scalar::C64(value.exp()),
            Scalar::C128(value) => Scalar::C128(value.exp()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the exponential of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Log for Scalar {
    /// Computes the elementwise natural logarithm of this [`Scalar`]. Only the floating-point and complex variants
    /// support the logarithm (the complex logarithm being the principal branch `ln(z)`); any other variant returns a
    /// [`TypeError`].
    fn log(&self) -> Result<Self, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Self::encode_low_precision_float(r#type, Self::decode_low_precision_float(r#type, bits).ln());
        }
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().ln())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().ln())),
            Scalar::F32(value) => Scalar::F32(value.ln()),
            Scalar::F64(value) => Scalar::F64(value.ln()),
            Scalar::C64(value) => Scalar::C64(value.ln()),
            Scalar::C128(value) => Scalar::C128(value.ln()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the logarithm of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Sqrt for Scalar {
    /// Computes the elementwise square root of this [`Scalar`]. Only the floating-point and complex variants support
    /// the square root (the complex square root being the principal branch `√z`); any other variant returns a
    /// [`TypeError`].
    fn sqrt(&self) -> Result<Self, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Self::encode_low_precision_float(r#type, Self::decode_low_precision_float(r#type, bits).sqrt());
        }
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().sqrt())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().sqrt())),
            Scalar::F32(value) => Scalar::F32(value.sqrt()),
            Scalar::F64(value) => Scalar::F64(value.sqrt()),
            Scalar::C64(value) => Scalar::C64(value.sqrt()),
            Scalar::C128(value) => Scalar::C128(value.sqrt()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the square root of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Rsqrt for Scalar {
    /// Computes the elementwise reciprocal square root of this [`Scalar`]. Only the floating-point and complex
    /// variants support the reciprocal square root (the complex form being the reciprocal of the principal branch
    /// `1/√z`); any other variant returns a [`TypeError`].
    fn rsqrt(&self) -> Result<Self, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Self::encode_low_precision_float(
                r#type,
                Self::decode_low_precision_float(r#type, bits).sqrt().recip(),
            );
        }
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().sqrt().recip())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().sqrt().recip())),
            Scalar::F32(value) => Scalar::F32(value.sqrt().recip()),
            Scalar::F64(value) => Scalar::F64(value.sqrt().recip()),
            Scalar::C64(value) => Scalar::C64(value.sqrt().inv()),
            Scalar::C128(value) => Scalar::C128(value.sqrt().inv()),
            other => {
                return Err(TypeError {
                    message: format!(
                        "cannot compute the reciprocal square root of a scalar of data type {}",
                        other.r#type(),
                    ),
                }
                .into());
            }
        })
    }
}

impl Tanh for Scalar {
    /// Computes the elementwise hyperbolic tangent of this [`Scalar`]. Only the floating-point and complex variants
    /// support the hyperbolic tangent (the complex form being the analytic continuation `tanh(z)`); any other variant
    /// returns a [`TypeError`].
    fn tanh(&self) -> Result<Self, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Self::encode_low_precision_float(r#type, Self::decode_low_precision_float(r#type, bits).tanh());
        }
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().tanh())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().tanh())),
            Scalar::F32(value) => Scalar::F32(value.tanh()),
            Scalar::F64(value) => Scalar::F64(value.tanh()),
            Scalar::C64(value) => Scalar::C64(value.tanh()),
            Scalar::C128(value) => Scalar::C128(value.tanh()),
            other => {
                return Err(TypeError {
                    message: format!(
                        "cannot compute the hyperbolic tangent of a scalar of data type {}",
                        other.r#type(),
                    ),
                }
                .into());
            }
        })
    }
}

impl Logistic for Scalar {
    /// Computes the elementwise logistic sigmoid `1 / (1 + e^{-x})` of this [`Scalar`]. Only the floating-point and
    /// complex variants support the logistic function (the complex form being the analytic continuation); any other
    /// variant returns a [`TypeError`].
    fn logistic(&self) -> Result<Self, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            let value = Self::decode_low_precision_float(r#type, bits);
            return Self::encode_low_precision_float(r#type, ((-value).exp() + 1.0).recip());
        }
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(((-value.to_f32()).exp() + 1.0).recip())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(((-value.to_f32()).exp() + 1.0).recip())),
            Scalar::F32(value) => Scalar::F32(((-value).exp() + 1.0).recip()),
            Scalar::F64(value) => Scalar::F64(((-value).exp() + 1.0).recip()),
            Scalar::C64(value) => Scalar::C64(((-value).exp() + 1.0).inv()),
            Scalar::C128(value) => Scalar::C128(((-value).exp() + 1.0).inv()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the logistic of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

/// Evaluates a polynomial at `x` with the provided coefficients ordered from the highest-degree term down to the
/// constant term, using Horner's scheme.
fn evaluate_polynomial(x: f64, coefficients: &[f64]) -> f64 {
    coefficients.iter().fold(0.0, |accumulator, coefficient| accumulator * x + coefficient)
}

/// Computes the Gauss error function `erf(x) = 2/√π · ∫₀ˣ e^{−t²} dt` in double precision.
///
/// This is the rational Chebyshev approximation from FDLIBM 5.3 (`s_erf.c`, developed at SunSoft and also used by
/// musl), split by argument magnitude: a short odd series below `2⁻²⁸`, a primary rational approximation on
/// `|x| < 0.84375`, a rational correction around `erf(1)` on `[0.84375, 1.25)`, two rational tail regimes evaluated
/// through the complementary function `1 − erfc(x)` on `[1.25, 6)`, and saturation to `±1` for `|x| ≥ 6` (where
/// `1 − erf(|x|) < 2⁻⁵⁶` is not representable next to one). The expected accuracy is about 1 ulp in double precision.
/// NaN inputs propagate and the sign symmetry `erf(−x) = −erf(x)` is exact, including for signed zeros. All
/// polynomial coefficient arrays below list the FDLIBM constants from the highest-degree term down to the constant
/// term, matching the [`evaluate_polynomial`] contract.
fn erf(x: f64) -> f64 {
    /// `erf(1)` rounded toward zero, used as the base value of the `[0.84375, 1.25)` regime.
    const ERX: f64 = 8.45062911510467529297e-01;

    /// Coefficient of the short odd series below `2⁻²⁸`, equal to `8 · (2/√π − 1)`.
    const EFX8: f64 = 1.02703333676410069053e+00;

    /// Numerator coefficients of the primary regime `erf(x) = x + x · PP(x²)/QQ(x²)` on `|x| < 0.84375`.
    const PP: [f64; 5] = [
        -2.37630166566501626084e-05,
        -5.77027029648944159157e-03,
        -2.84817495755985104766e-02,
        -3.25042107247001499370e-01,
        1.28379167095512558561e-01,
    ];

    /// Denominator coefficients of the primary regime on `|x| < 0.84375`.
    const QQ: [f64; 6] = [
        -3.96022827877536812320e-06,
        1.32494738004321644526e-04,
        5.08130628187576562776e-03,
        6.50222499887672944485e-02,
        3.97917223959155352819e-01,
        1.0,
    ];

    /// Numerator coefficients of the regime `erf(x) = sign(x) · (ERX + PA(|x|−1)/QA(|x|−1))` on `[0.84375, 1.25)`.
    const PA: [f64; 7] = [
        -2.16637559486879084300e-03,
        3.54783043256182359371e-02,
        -1.10894694282396677476e-01,
        3.18346619901161753674e-01,
        -3.72207876035701323847e-01,
        4.14856118683748331666e-01,
        -2.36211856075265944077e-03,
    ];

    /// Denominator coefficients of the `[0.84375, 1.25)` regime.
    const QA: [f64; 7] = [
        1.19844998467991074170e-02,
        1.36370839120290507362e-02,
        1.26171219808761642112e-01,
        7.18286544141962662868e-02,
        5.40397917702171048937e-01,
        1.06420880400844228286e-01,
        1.0,
    ];

    /// Numerator coefficients of the complementary-function tail on `[1.25, 1/0.35)`, in the variable `1/x²`.
    const RA: [f64; 8] = [
        -9.81432934416914548592e+00,
        -8.12874355063065934246e+01,
        -1.84605092906711035994e+02,
        -1.62396669462573470355e+02,
        -6.23753324503260060396e+01,
        -1.05586262253232909814e+01,
        -6.93858572707181764372e-01,
        -9.86494403484714822705e-03,
    ];

    /// Denominator coefficients of the complementary-function tail on `[1.25, 1/0.35)`, in the variable `1/x²`.
    const SA: [f64; 9] = [
        -6.04244152148580987438e-02,
        6.57024977031928170135e+00,
        1.08635005541779435134e+02,
        4.29008140027567833386e+02,
        6.45387271733267880336e+02,
        4.34565877475229228821e+02,
        1.37657754143519042600e+02,
        1.96512716674392571292e+01,
        1.0,
    ];

    /// Numerator coefficients of the complementary-function tail on `[1/0.35, 6)`, in the variable `1/x²`.
    const RB: [f64; 7] = [
        -4.83519191608651397019e+02,
        -1.02509513161107724954e+03,
        -6.37566443368389627722e+02,
        -1.60636384855821916062e+02,
        -1.77579549177547519889e+01,
        -7.99283237680523006574e-01,
        -9.86494292470009928597e-03,
    ];

    /// Denominator coefficients of the complementary-function tail on `[1/0.35, 6)`, in the variable `1/x²`.
    const SB: [f64; 8] = [
        -2.24409524465858183362e+01,
        4.74528541206955367215e+02,
        2.55305040643316442583e+03,
        3.19985821950859553908e+03,
        1.53672958608443695994e+03,
        3.25792512996573918826e+02,
        3.03380607434824582924e+01,
        1.0,
    ];

    if x.is_nan() {
        return x;
    }
    let negative = x.is_sign_negative();
    let magnitude = x.abs();
    if magnitude < 0.84375 {
        if magnitude < 3.725290298461914e-09 {
            // For |x| < 2⁻²⁸ the series truncates to its leading odd term `x · 2/√π`, evaluated in a
            // scaled form that avoids intermediate underflow and preserves signed zeros.
            return 0.125 * (8.0 * x + EFX8 * x);
        }
        let squared = x * x;
        return x + x * (evaluate_polynomial(squared, &PP) / evaluate_polynomial(squared, &QQ));
    }
    if magnitude < 1.25 {
        let shifted = magnitude - 1.0;
        let correction = evaluate_polynomial(shifted, &PA) / evaluate_polynomial(shifted, &QA);
        return if negative { -ERX - correction } else { ERX + correction };
    }
    if magnitude >= 6.0 {
        // Covers |x| ≥ 6 and infinities: 1 − erf(6) < 2⁻⁵⁶ already rounds to zero next to one.
        return if negative { -1.0 } else { 1.0 };
    }
    // On [1.25, 6) the error function is evaluated through its complement as `erf(x) = sign(x) · (1 − erfc(|x|))`
    // with `erfc(y) = exp(−z² − 0.5625) · exp((z − y)(z + y) + R(1/y²)/S(1/y²)) / y`, where `z` is `y` with the low
    // half of its mantissa cleared so that `z²` is exact and the argument reduction stays accurate.
    let inverse_squared = 1.0 / (magnitude * magnitude);
    let quotient = if magnitude < 1.0 / 0.35 {
        evaluate_polynomial(inverse_squared, &RA) / evaluate_polynomial(inverse_squared, &SA)
    } else {
        evaluate_polynomial(inverse_squared, &RB) / evaluate_polynomial(inverse_squared, &SB)
    };
    let truncated = f64::from_bits(magnitude.to_bits() & 0xffff_ffff_0000_0000);
    let complement = (-truncated * truncated - 0.5625).exp()
        * ((truncated - magnitude) * (truncated + magnitude) + quotient).exp()
        / magnitude;
    if negative { complement - 1.0 } else { 1.0 - complement }
}

impl Erf for Scalar {
    /// Computes the elementwise Gauss error function of this [`Scalar`]. Only the real floating-point variants
    /// support the error function; any other variant returns a [`TypeError`]. Every variant evaluates through the
    /// double-precision [`erf`] approximation and rounds the result to its own precision.
    fn erf(&self) -> Result<Self, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Self::encode_low_precision_float(r#type, erf(Self::decode_low_precision_float(r#type, bits)));
        }
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f64(erf(value.to_f64()))),
            Scalar::F16(value) => Scalar::F16(f16::from_f64(erf(value.to_f64()))),
            Scalar::F32(value) => Scalar::F32(erf(f64::from(*value)) as f32),
            Scalar::F64(value) => Scalar::F64(erf(*value)),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the error function of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Pow for Scalar {
    /// Computes the elementwise power `self^exponent` for this [`Scalar`]. Floating-point and complex operands are
    /// promoted to a common data type (the complex power being the principal value `exp(y · log(x))`); any other
    /// combination returns a [`TypeError`].
    fn pow(&self, exponent: &Self) -> Result<Self, ProgramError> {
        check_types!(@float, "pow", [self.r#type().into_owned(), exponent.r#type().into_owned()]);
        let (base, exponent) = promote_scalar_arithmetic_operands(self, exponent, "pow")?;
        if let (Some((left_type, left_bits)), Some((right_type, right_bits))) =
            (base.low_precision_float_parts(), exponent.low_precision_float_parts())
        {
            return Self::encode_low_precision_float(
                left_type,
                Self::decode_low_precision_float(left_type, left_bits)
                    .powf(Self::decode_low_precision_float(right_type, right_bits)),
            );
        }
        Ok(match (base, exponent) {
            (Scalar::BF16(base), Scalar::BF16(exponent)) => {
                Scalar::BF16(bf16::from_f32(base.to_f32().powf(exponent.to_f32())))
            }
            (Scalar::F16(base), Scalar::F16(exponent)) => {
                Scalar::F16(f16::from_f32(base.to_f32().powf(exponent.to_f32())))
            }
            (Scalar::F32(base), Scalar::F32(exponent)) => Scalar::F32(base.powf(exponent)),
            (Scalar::F64(base), Scalar::F64(exponent)) => Scalar::F64(base.powf(exponent)),
            (Scalar::C64(base), Scalar::C64(exponent)) => Scalar::C64(base.powc(exponent)),
            (Scalar::C128(base), Scalar::C128(exponent)) => Scalar::C128(base.powc(exponent)),
            (base, exponent) => {
                return Err(TypeError {
                    message: format!(
                        "cannot compute the power of scalars of data types {} and {}",
                        base.r#type(),
                        exponent.r#type(),
                    ),
                }
                .into());
            }
        })
    }
}

impl Sign for Scalar {
    /// Computes the elementwise sign of this [`Scalar`]: `-1`, `0`, or `1` for the signed-integer variants; `-1.0`
    /// or `1.0` for floating-point values away from zero, with signed zeros and NaNs passing through unchanged; and
    /// `z / |z|` for the complex variants, with `0` mapping to itself. Unsigned-integer, Boolean, and other variants
    /// return a [`TypeError`].
    fn sign(&self) -> Result<Self, ProgramError> {
        /// Computes the floating-point sign, preserving signed zeros and NaNs.
        fn float_sign(value: f64) -> f64 {
            if value.is_nan() || value == 0.0 { value } else { value.signum() }
        }

        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Self::encode_low_precision_float(
                r#type,
                float_sign(Self::decode_low_precision_float(r#type, bits)),
            );
        }
        Ok(match self {
            Scalar::I8(value) => Scalar::I8(value.signum()),
            Scalar::I16(value) => Scalar::I16(value.signum()),
            Scalar::I32(value) => Scalar::I32(value.signum()),
            Scalar::I64(value) => Scalar::I64(value.signum()),
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(float_sign(value.to_f64()) as f32)),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(float_sign(value.to_f64()) as f32)),
            Scalar::F32(value) => Scalar::F32(float_sign(f64::from(*value)) as f32),
            Scalar::F64(value) => Scalar::F64(float_sign(*value)),
            Scalar::C64(value) => {
                let norm = value.norm();
                Scalar::C64(if norm == 0.0 { *value } else { value / norm })
            }
            Scalar::C128(value) => {
                let norm = value.norm();
                Scalar::C128(if norm == 0.0 { *value } else { value / norm })
            }
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the sign of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Floor for Scalar {
    /// Computes the elementwise floor of this [`Scalar`], rounding toward negative infinity. Only the real
    /// floating-point variants support the floor; any other variant returns a [`TypeError`].
    fn floor(&self) -> Result<Self, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Self::encode_low_precision_float(r#type, Self::decode_low_precision_float(r#type, bits).floor());
        }
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().floor())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().floor())),
            Scalar::F32(value) => Scalar::F32(value.floor()),
            Scalar::F64(value) => Scalar::F64(value.floor()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the floor of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Ceil for Scalar {
    /// Computes the elementwise ceiling of this [`Scalar`], rounding toward positive infinity. Only the real
    /// floating-point variants support the ceiling; any other variant returns a [`TypeError`].
    fn ceil(&self) -> Result<Self, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Self::encode_low_precision_float(r#type, Self::decode_low_precision_float(r#type, bits).ceil());
        }
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().ceil())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().ceil())),
            Scalar::F32(value) => Scalar::F32(value.ceil()),
            Scalar::F64(value) => Scalar::F64(value.ceil()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the ceiling of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Round for Scalar {
    /// Computes the elementwise rounding of this [`Scalar`] to the nearest integer, resolving ties toward the
    /// nearest even integer. Only the real floating-point variants support rounding; any other variant returns a
    /// [`TypeError`].
    fn round(&self) -> Result<Self, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Self::encode_low_precision_float(
                r#type,
                Self::decode_low_precision_float(r#type, bits).round_ties_even(),
            );
        }
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().round_ties_even())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().round_ties_even())),
            Scalar::F32(value) => Scalar::F32(value.round_ties_even()),
            Scalar::F64(value) => Scalar::F64(value.round_ties_even()),
            other => {
                return Err(
                    TypeError { message: format!("cannot round a scalar of data type {}", other.r#type()) }.into()
                );
            }
        })
    }
}

/// Selects the maximum of two floating-point values with NaN propagation and `-0.0` ordering below `+0.0`, resolving
/// exact ties toward `left`.
fn maximum_float(left: f64, right: f64) -> f64 {
    if left.is_nan() {
        left
    } else if right.is_nan() {
        right
    } else {
        match left.total_cmp(&right) {
            Ordering::Less => right,
            _ => left,
        }
    }
}

/// Selects the minimum of two floating-point values with NaN propagation and `-0.0` ordering below `+0.0`, resolving
/// exact ties toward `left`.
fn minimum_float(left: f64, right: f64) -> f64 {
    if left.is_nan() {
        left
    } else if right.is_nan() {
        right
    } else {
        match left.total_cmp(&right) {
            Ordering::Greater => right,
            _ => left,
        }
    }
}

impl Maximum for Scalar {
    /// Computes the elementwise maximum of two [`Scalar`]s, promoting them to a common data type. Only the real
    /// (non-complex) numeric variants are supported. For floating-point operands, NaNs propagate and `-0.0` orders
    /// below `+0.0`; any other combination returns a [`TypeError`].
    fn maximum(&self, right: &Self) -> Result<Self, ProgramError> {
        check_types!(@numeric @real, "maximum", [self.r#type().into_owned(), right.r#type().into_owned()]);
        let (left, right) = promote_scalar_arithmetic_operands(self, right, "maximum")?;
        if let (Some((left_type, left_bits)), Some((right_type, right_bits))) =
            (left.low_precision_float_parts(), right.low_precision_float_parts())
        {
            return Self::encode_low_precision_float(
                left_type,
                maximum_float(
                    Self::decode_low_precision_float(left_type, left_bits),
                    Self::decode_low_precision_float(right_type, right_bits),
                ),
            );
        }
        Ok(match (left, right) {
            (Scalar::I8(left), Scalar::I8(right)) => Scalar::I8(left.max(right)),
            (Scalar::I16(left), Scalar::I16(right)) => Scalar::I16(left.max(right)),
            (Scalar::I32(left), Scalar::I32(right)) => Scalar::I32(left.max(right)),
            (Scalar::I64(left), Scalar::I64(right)) => Scalar::I64(left.max(right)),
            (Scalar::U8(left), Scalar::U8(right)) => Scalar::U8(left.max(right)),
            (Scalar::U16(left), Scalar::U16(right)) => Scalar::U16(left.max(right)),
            (Scalar::U32(left), Scalar::U32(right)) => Scalar::U32(left.max(right)),
            (Scalar::U64(left), Scalar::U64(right)) => Scalar::U64(left.max(right)),
            (Scalar::BF16(left), Scalar::BF16(right)) => {
                Scalar::BF16(bf16::from_f64(maximum_float(left.to_f64(), right.to_f64())))
            }
            (Scalar::F16(left), Scalar::F16(right)) => {
                Scalar::F16(f16::from_f64(maximum_float(left.to_f64(), right.to_f64())))
            }
            (Scalar::F32(left), Scalar::F32(right)) => {
                Scalar::F32(maximum_float(f64::from(left), f64::from(right)) as f32)
            }
            (Scalar::F64(left), Scalar::F64(right)) => Scalar::F64(maximum_float(left, right)),
            (left, right) => {
                return Err(TypeError {
                    message: format!(
                        "cannot compute the maximum of scalars of data types {} and {}",
                        left.r#type(),
                        right.r#type(),
                    ),
                }
                .into());
            }
        })
    }
}

impl Minimum for Scalar {
    /// Computes the elementwise minimum of two [`Scalar`]s, promoting them to a common data type. Only the real
    /// (non-complex) numeric variants are supported. For floating-point operands, NaNs propagate and `-0.0` orders
    /// below `+0.0`; any other combination returns a [`TypeError`].
    fn minimum(&self, right: &Self) -> Result<Self, ProgramError> {
        check_types!(@numeric @real, "minimum", [self.r#type().into_owned(), right.r#type().into_owned()]);
        let (left, right) = promote_scalar_arithmetic_operands(self, right, "minimum")?;
        if let (Some((left_type, left_bits)), Some((right_type, right_bits))) =
            (left.low_precision_float_parts(), right.low_precision_float_parts())
        {
            return Self::encode_low_precision_float(
                left_type,
                minimum_float(
                    Self::decode_low_precision_float(left_type, left_bits),
                    Self::decode_low_precision_float(right_type, right_bits),
                ),
            );
        }
        Ok(match (left, right) {
            (Scalar::I8(left), Scalar::I8(right)) => Scalar::I8(left.min(right)),
            (Scalar::I16(left), Scalar::I16(right)) => Scalar::I16(left.min(right)),
            (Scalar::I32(left), Scalar::I32(right)) => Scalar::I32(left.min(right)),
            (Scalar::I64(left), Scalar::I64(right)) => Scalar::I64(left.min(right)),
            (Scalar::U8(left), Scalar::U8(right)) => Scalar::U8(left.min(right)),
            (Scalar::U16(left), Scalar::U16(right)) => Scalar::U16(left.min(right)),
            (Scalar::U32(left), Scalar::U32(right)) => Scalar::U32(left.min(right)),
            (Scalar::U64(left), Scalar::U64(right)) => Scalar::U64(left.min(right)),
            (Scalar::BF16(left), Scalar::BF16(right)) => {
                Scalar::BF16(bf16::from_f64(minimum_float(left.to_f64(), right.to_f64())))
            }
            (Scalar::F16(left), Scalar::F16(right)) => {
                Scalar::F16(f16::from_f64(minimum_float(left.to_f64(), right.to_f64())))
            }
            (Scalar::F32(left), Scalar::F32(right)) => {
                Scalar::F32(minimum_float(f64::from(left), f64::from(right)) as f32)
            }
            (Scalar::F64(left), Scalar::F64(right)) => Scalar::F64(minimum_float(left, right)),
            (left, right) => {
                return Err(TypeError {
                    message: format!(
                        "cannot compute the minimum of scalars of data types {} and {}",
                        left.r#type(),
                        right.r#type(),
                    ),
                }
                .into());
            }
        })
    }
}

impl Remainder for Scalar {
    /// Computes the elementwise remainder of this [`Scalar`] (the dividend) and `right` (the divisor), with the
    /// result taking the sign of the dividend (truncation semantics). Integer and floating-point operands are
    /// promoted to a common data type; integer remainders with a zero divisor return a [`TypeError`], the minimum
    /// signed-integer dividend with a `-1` divisor wraps to `0`, and any other combination returns a [`TypeError`].
    fn remainder(&self, right: &Self) -> Result<Self, ProgramError> {
        check_types!(@numeric @real, "remainder", [self.r#type().into_owned(), right.r#type().into_owned()]);
        let (left, right) = promote_scalar_arithmetic_operands(self, right, "remainder")?;
        if let (Some((left_type, left_bits)), Some((right_type, right_bits))) =
            (left.low_precision_float_parts(), right.low_precision_float_parts())
        {
            return Self::encode_low_precision_float(
                left_type,
                Self::decode_low_precision_float(left_type, left_bits)
                    % Self::decode_low_precision_float(right_type, right_bits),
            );
        }

        macro_rules! integer_remainder {
            ($left:expr, $right:expr, $variant:ident, $data_type:ident) => {{
                if $right == 0 {
                    return Err(TypeError {
                        message: format!(
                            "cannot compute the remainder of an integer scalar of data type {} with a zero divisor",
                            DataType::$data_type,
                        ),
                    }
                    .into());
                }
                Scalar::$variant($left.wrapping_rem($right))
            }};
        }

        Ok(match (left, right) {
            (Scalar::I8(left), Scalar::I8(right)) => integer_remainder!(left, right, I8, I8),
            (Scalar::I16(left), Scalar::I16(right)) => integer_remainder!(left, right, I16, I16),
            (Scalar::I32(left), Scalar::I32(right)) => integer_remainder!(left, right, I32, I32),
            (Scalar::I64(left), Scalar::I64(right)) => integer_remainder!(left, right, I64, I64),
            (Scalar::U8(left), Scalar::U8(right)) => integer_remainder!(left, right, U8, U8),
            (Scalar::U16(left), Scalar::U16(right)) => integer_remainder!(left, right, U16, U16),
            (Scalar::U32(left), Scalar::U32(right)) => integer_remainder!(left, right, U32, U32),
            (Scalar::U64(left), Scalar::U64(right)) => integer_remainder!(left, right, U64, U64),
            (Scalar::BF16(left), Scalar::BF16(right)) => Scalar::BF16(bf16::from_f32(left.to_f32() % right.to_f32())),
            (Scalar::F16(left), Scalar::F16(right)) => Scalar::F16(f16::from_f32(left.to_f32() % right.to_f32())),
            (Scalar::F32(left), Scalar::F32(right)) => Scalar::F32(left % right),
            (Scalar::F64(left), Scalar::F64(right)) => Scalar::F64(left % right),
            (left, right) => {
                return Err(TypeError {
                    message: format!(
                        "cannot compute the remainder of scalars of data types {} and {}",
                        left.r#type(),
                        right.r#type(),
                    ),
                }
                .into());
            }
        })
    }
}

impl Not for Scalar {
    /// Computes the elementwise negation of this [`Scalar`]. A Boolean scalar negates logically and an integer scalar
    /// negates bitwise (the two semantics that StableHLO's `not` operation also serves); any other variant returns a
    /// [`TypeError`].
    fn not(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::Bool(value) => Scalar::Bool(!value),
            Scalar::I8(value) => Scalar::I8(!value),
            Scalar::I16(value) => Scalar::I16(!value),
            Scalar::I32(value) => Scalar::I32(!value),
            Scalar::I64(value) => Scalar::I64(!value),
            Scalar::U8(value) => Scalar::U8(!value),
            Scalar::U16(value) => Scalar::U16(!value),
            Scalar::U32(value) => Scalar::U32(!value),
            Scalar::U64(value) => Scalar::U64(!value),
            other => {
                return Err(TypeError {
                    message: format!("cannot apply `not` to a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl std::ops::Not for Scalar {
    type Output = Scalar;

    #[inline]
    fn not(self) -> Scalar {
        Not::not(&self).unwrap_or_else(|error| panic!("{error}"))
    }
}

/// Implements a fallible binary logical capability (e.g., [`And`]) together with its panicking [`std::ops`]
/// counterpart (e.g., [`std::ops::BitAnd`]) for [`Scalar`]. Boolean operands combine logically and same-variant
/// integer operands combine bitwise (the two semantics that StableHLO's logical operations also serve); any other
/// combination returns a [`TypeError`].
macro_rules! impl_binary_logical_for_scalar {
    ($trait:ident, $std_trait:ident, $method:ident, $std_method:ident, $operator:tt) => {
        impl $trait for Scalar {
            #[inline]
            fn $method(&self, rhs: &Scalar) -> Result<Scalar, ProgramError> {
                Ok(match (*self, *rhs) {
                    (Scalar::Bool(left), Scalar::Bool(right)) => Scalar::Bool(left $operator right),
                    (Scalar::I8(left), Scalar::I8(right)) => Scalar::I8(left $operator right),
                    (Scalar::I16(left), Scalar::I16(right)) => Scalar::I16(left $operator right),
                    (Scalar::I32(left), Scalar::I32(right)) => Scalar::I32(left $operator right),
                    (Scalar::I64(left), Scalar::I64(right)) => Scalar::I64(left $operator right),
                    (Scalar::U8(left), Scalar::U8(right)) => Scalar::U8(left $operator right),
                    (Scalar::U16(left), Scalar::U16(right)) => Scalar::U16(left $operator right),
                    (Scalar::U32(left), Scalar::U32(right)) => Scalar::U32(left $operator right),
                    (Scalar::U64(left), Scalar::U64(right)) => Scalar::U64(left $operator right),
                    (left, right) => {
                        return Err(TypeError {
                            message: format!(
                                "cannot apply `{}` to scalars of data types {} and {}",
                                stringify!($method),
                                left.r#type(),
                                right.r#type(),
                            ),
                        }
                        .into());
                    }
                })
            }
        }

        impl std::ops::$std_trait for Scalar {
            type Output = Scalar;

            #[inline]
            fn $std_method(self, rhs: Scalar) -> Scalar {
                $trait::$method(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
            }
        }
    };
}

impl_binary_logical_for_scalar!(And, BitAnd, and, bitand, &);
impl_binary_logical_for_scalar!(Or, BitOr, or, bitor, |);
impl_binary_logical_for_scalar!(Xor, BitXor, xor, bitxor, ^);

// The `Complex` construction capability is implemented with a path-qualified trait name because this module already
// uses `num_complex::Complex` pervasively as the complex payload type.
impl crate::operations::complex::Complex for Scalar {
    /// Constructs a complex [`Scalar`] from this value as the real part and `imaginary` as the imaginary part. Only
    /// same-precision `f32` and `f64` part pairs are supported; any other combination returns a [`TypeError`].
    fn complex(&self, imaginary: &Self) -> Result<Self, ProgramError> {
        Ok(match (*self, *imaginary) {
            (Scalar::F32(real), Scalar::F32(imaginary)) => Scalar::C64(Complex::new(real, imaginary)),
            (Scalar::F64(real), Scalar::F64(imaginary)) => Scalar::C128(Complex::new(real, imaginary)),
            (real, imaginary) => {
                return Err(TypeError {
                    message: format!(
                        "cannot construct a complex scalar from parts of data types {} and {}",
                        real.r#type(),
                        imaginary.r#type(),
                    ),
                }
                .into());
            }
        })
    }
}

impl Conjugate for Scalar {
    /// Computes the complex conjugate of this [`Scalar`]. Only the complex variants support conjugation; any other
    /// variant returns a [`TypeError`].
    fn conjugate(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::C64(value) => Scalar::C64(value.conj()),
            Scalar::C128(value) => Scalar::C128(value.conj()),
            other => {
                return Err(TypeError {
                    message: format!("cannot conjugate a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Real for Scalar {
    /// Extracts the real part of this complex [`Scalar`]. Only the complex variants support the extraction; any other
    /// variant returns a [`TypeError`].
    fn real(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::C64(value) => Scalar::F32(value.re),
            Scalar::C128(value) => Scalar::F64(value.re),
            other => {
                return Err(TypeError {
                    message: format!("cannot extract the real part of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Imaginary for Scalar {
    /// Extracts the imaginary part of this complex [`Scalar`]. Only the complex variants support the extraction; any
    /// other variant returns a [`TypeError`].
    fn imaginary(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::C64(value) => Scalar::F32(value.im),
            Scalar::C128(value) => Scalar::F64(value.im),
            other => {
                return Err(TypeError {
                    message: format!("cannot extract the imaginary part of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Compare for Scalar {
    type Output = Scalar;

    /// Compares two equal-[`DataType`] [`Scalar`]s and returns the Boolean result as an honestly Boolean-typed
    /// [`Scalar::Bool`], never a numeric variant. Mismatched variants return a [`TypeError`]. Complex scalars are
    /// unordered, so they support only the equality directions and return a [`TypeError`] for ordered ones.
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Self::Output, ProgramError> {
        /// Evaluates a comparison `direction` against an optional `ordering`, where `None` (an unordered pair, e.g.,
        /// one involving a NaN) satisfies only the `NotEqual` direction.
        fn evaluate(ordering: Option<Ordering>, direction: ComparisonDirection) -> bool {
            match direction {
                ComparisonDirection::Equal => ordering == Some(Ordering::Equal),
                ComparisonDirection::NotEqual => ordering != Some(Ordering::Equal),
                ComparisonDirection::LessThan => ordering == Some(Ordering::Less),
                ComparisonDirection::LessThanOrEqual => matches!(ordering, Some(Ordering::Less | Ordering::Equal)),
                ComparisonDirection::GreaterThan => ordering == Some(Ordering::Greater),
                ComparisonDirection::GreaterThanOrEqual => {
                    matches!(ordering, Some(Ordering::Greater | Ordering::Equal))
                }
            }
        }

        if let (Some((left_type, left_bits)), Some((right_type, right_bits))) =
            (self.low_precision_float_parts(), rhs.low_precision_float_parts())
        {
            if left_type != right_type {
                return Err(TypeError {
                    message: format!("cannot compare scalars of data types {left_type} and {right_type}"),
                }
                .into());
            }
            let ordering = Self::decode_low_precision_float(left_type, left_bits)
                .partial_cmp(&Self::decode_low_precision_float(right_type, right_bits));
            return Ok(Scalar::Bool(evaluate(ordering, direction)));
        }
        if matches!(self, Scalar::Token) || matches!(rhs, Scalar::Token) {
            return Err(TypeError { message: "cannot compare token scalars".to_string() }.into());
        }
        if let (Scalar::C64(_), Scalar::C64(_)) | (Scalar::C128(_), Scalar::C128(_)) = (self, rhs) {
            return match direction {
                ComparisonDirection::Equal => Ok(Scalar::Bool(self == rhs)),
                ComparisonDirection::NotEqual => Ok(Scalar::Bool(self != rhs)),
                _ => Err(TypeError {
                    message: format!(
                        "cannot apply an ordered comparison to unordered complex scalars of data type {}",
                        self.r#type(),
                    ),
                }
                .into()),
            };
        }
        let ordering = match (self, rhs) {
            (Scalar::Bool(left), Scalar::Bool(right)) => left.partial_cmp(right),
            (Scalar::I8(left), Scalar::I8(right)) => left.partial_cmp(right),
            (Scalar::I16(left), Scalar::I16(right)) => left.partial_cmp(right),
            (Scalar::I32(left), Scalar::I32(right)) => left.partial_cmp(right),
            (Scalar::I64(left), Scalar::I64(right)) => left.partial_cmp(right),
            (Scalar::U8(left), Scalar::U8(right)) => left.partial_cmp(right),
            (Scalar::U16(left), Scalar::U16(right)) => left.partial_cmp(right),
            (Scalar::U32(left), Scalar::U32(right)) => left.partial_cmp(right),
            (Scalar::U64(left), Scalar::U64(right)) => left.partial_cmp(right),
            (Scalar::BF16(left), Scalar::BF16(right)) => left.partial_cmp(right),
            (Scalar::F16(left), Scalar::F16(right)) => left.partial_cmp(right),
            (Scalar::F32(left), Scalar::F32(right)) => left.partial_cmp(right),
            (Scalar::F64(left), Scalar::F64(right)) => left.partial_cmp(right),
            (left, right) => {
                return Err(TypeError {
                    message: format!("cannot compare scalars of data types {} and {}", left.r#type(), right.r#type()),
                }
                .into());
            }
        };
        Ok(Scalar::Bool(evaluate(ordering, direction)))
    }
}

impl Select for Scalar {
    /// Selects between `on_true` and `on_false` after decoding the in-band `condition` through [`BooleanLike`],
    /// mirroring the broadcasting [`SelectOperation`] type-inference contract. The selected branch is promoted to the
    /// promotion of the two branch data types, so `select(condition, f32, f64)` yields an `f64` like `jnp.where`.
    #[inline]
    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let condition = condition.boolean()?;
        let target = DataType::promoted(&[on_true.r#type().into_owned(), on_false.r#type().into_owned()])
            .map_err(|error| TypeError { message: error.to_string() })?;
        let selected = if condition { on_true } else { on_false };
        selected.convert_element_type(target)
    }
}

impl BooleanLike for Scalar {
    #[inline]
    fn as_boolean(&self) -> Self {
        match self.boolean() {
            Ok(value) => Scalar::Bool(value),
            Err(_) => *self,
        }
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        if let Some((r#type, bits)) = self.low_precision_float_parts() {
            return Ok(Self::decode_low_precision_float(r#type, bits) != 0.0);
        }
        Ok(match self {
            Scalar::Token => {
                return Err(TypeError { message: "cannot convert a token scalar to a Boolean".to_string() }.into());
            }
            Scalar::Zero => {
                return Err(TypeError { message: "cannot convert a zero-space scalar to a Boolean".to_string() }.into());
            }
            Scalar::Bool(value) => *value,
            Scalar::I8(value) => *value != 0,
            Scalar::I16(value) => *value != 0,
            Scalar::I32(value) => *value != 0,
            Scalar::I64(value) => *value != 0,
            Scalar::U8(value) => *value != 0,
            Scalar::U16(value) => *value != 0,
            Scalar::U32(value) => *value != 0,
            Scalar::U64(value) => *value != 0,
            // The low-precision variants were already handled through `low_precision_float_parts` before this
            // match, which must nevertheless remain exhaustive.
            Scalar::F4E2M1FN(_)
            | Scalar::F8E3M4(_)
            | Scalar::F8E4M3(_)
            | Scalar::F8E4M3FN(_)
            | Scalar::F8E4M3FNUZ(_)
            | Scalar::F8E4M3B11FNUZ(_)
            | Scalar::F8E5M2(_)
            | Scalar::F8E5M2FNUZ(_)
            | Scalar::F8E8M0FNU(_) => unreachable!("handled before the match"),
            Scalar::BF16(value) => *value != bf16::ZERO,
            Scalar::F16(value) => *value != f16::ZERO,
            Scalar::F32(value) => *value != 0.0,
            Scalar::F64(value) => *value != 0.0,
            Scalar::C64(value) => value.re != 0.0 || value.im != 0.0,
            Scalar::C128(value) => value.re != 0.0 || value.im != 0.0,
        })
    }
}

// A `Scalar` predicate is always rank-0 and so the scalar `WhilePredicate` defaults (its own truth value decides
// continuation, and a true predicate takes the candidate wholesale) are exactly its semantics.
impl WhilePredicate for Scalar {}

impl ConvertElementType for Scalar {
    fn convert_element_type(&self, target: DataType) -> Result<Self, ProgramError> {
        macro_rules! convert_real {
            ($value:expr) => {{
                let value = $value;
                match target {
                    DataType::Boolean => Scalar::Bool(value as f64 != 0.0),
                    DataType::I8 => Scalar::I8(value as i8),
                    DataType::I16 => Scalar::I16(value as i16),
                    DataType::I32 => Scalar::I32(value as i32),
                    DataType::I64 => Scalar::I64(value as i64),
                    DataType::U8 => Scalar::U8(value as u8),
                    DataType::U16 => Scalar::U16(value as u16),
                    DataType::U32 => Scalar::U32(value as u32),
                    DataType::U64 => Scalar::U64(value as u64),
                    DataType::BF16 => Scalar::BF16(bf16::from_f64(value as f64)),
                    DataType::F16 => Scalar::F16(f16::from_f64(value as f64)),
                    DataType::F32 => Scalar::F32(value as f32),
                    DataType::F64 => Scalar::F64(value as f64),
                    DataType::C64 => Scalar::C64(Complex::new(value as f32, 0.0)),
                    DataType::C128 => Scalar::C128(Complex::new(value as f64, 0.0)),
                    data_type
                        if matches!(
                            data_type,
                            DataType::F4E2M1FN
                                | DataType::F6E2M3FN
                                | DataType::F6E3M2FN
                                | DataType::F8E3M4
                                | DataType::F8E4M3
                                | DataType::F8E4M3FN
                                | DataType::F8E4M3FNUZ
                                | DataType::F8E4M3B11FNUZ
                                | DataType::F8E5M2
                                | DataType::F8E5M2FNUZ
                                | DataType::F8E8M0FNU
                        ) =>
                    {
                        return Self::encode_low_precision_float(data_type, value as f64);
                    }
                    other => {
                        return Err(TypeError {
                            message: format!("cannot convert scalar to unsupported data type {other}"),
                        }
                        .into());
                    }
                }
            }};
        }

        let source = self.r#type().into_owned();
        if source == target {
            return Ok(*self);
        }
        if source == DataType::Token || target == DataType::Token {
            return Err(
                TypeError { message: "cannot convert values to or from the token data type".to_string() }.into()
            );
        }
        if source == DataType::Zero || target == DataType::Zero {
            return Err(TypeError { message: "cannot convert values to or from the zero data type".to_string() }.into());
        }
        let converted = if let Some((data_type, bits)) = self.low_precision_float_parts() {
            convert_real!(Self::decode_low_precision_float(data_type, bits))
        } else {
            match self {
                Scalar::Bool(value) => convert_real!(u8::from(*value)),
                Scalar::I8(value) => convert_real!(*value),
                Scalar::I16(value) => convert_real!(*value),
                Scalar::I32(value) => convert_real!(*value),
                Scalar::I64(value) => convert_real!(*value),
                Scalar::U8(value) => convert_real!(*value),
                Scalar::U16(value) => convert_real!(*value),
                Scalar::U32(value) => convert_real!(*value),
                Scalar::U64(value) => convert_real!(*value),
                Scalar::BF16(value) => convert_real!(value.to_f64()),
                Scalar::F16(value) => convert_real!(value.to_f64()),
                Scalar::F32(value) => convert_real!(*value),
                Scalar::F64(value) => convert_real!(*value),
                Scalar::C64(value) => match target {
                    DataType::C64 => return Ok(*self),
                    DataType::C128 => return Ok(Scalar::C128(Complex::new(value.re as f64, value.im as f64))),
                    DataType::Boolean => return Ok(Scalar::Bool(value.re != 0.0 || value.im != 0.0)),
                    _ => convert_real!(value.re),
                },
                Scalar::C128(value) => match target {
                    DataType::C64 => return Ok(Scalar::C64(Complex::new(value.re as f32, value.im as f32))),
                    DataType::C128 => return Ok(*self),
                    DataType::Boolean => return Ok(Scalar::Bool(value.re != 0.0 || value.im != 0.0)),
                    _ => convert_real!(value.re),
                },
                other => {
                    return Err(TypeError {
                        message: format!("cannot convert scalar of data type {} to {target}", other.r#type()),
                    }
                    .into());
                }
            }
        };
        Ok(converted)
    }
}

impl StopGradient for Scalar {
    /// Returns this [`Scalar`] unchanged while marking it as a constant for differentiation purposes.
    #[inline]
    fn stop_gradient(&self) -> Self {
        *self
    }
}

impl Tag for Scalar {
    /// Returns this [`Scalar`] unchanged. Tagging is the identity on concrete values; the tag only matters when staging
    /// through a [`Tracer`](crate::tracing::Tracer).
    #[inline]
    fn tag(self, _key: &str) -> Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::{Context, StagingContext};
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder};
    use crate::tracing::Trace;
    use crate::tracing_v2::ForwardModeDifferentiate;

    use super::*;

    /// Builds a carry-only scalar body program that maps `[carry]` to `[carry + carry]`.
    fn scalar_doubling_body() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let carry = builder.add_input(DataType::F64);
        let doubled = builder.add_instruction(AddOperation, Vec::new(), vec![carry, carry]).unwrap()[0];
        builder.build(vec![doubled], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a scalar while condition that maps `[carry]` to `[carry < 8]`.
    fn scalar_less_than_eight_condition() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let carry = builder.add_input(DataType::F64);
        let eight = builder.add_constant(Scalar::from(8.0));
        let predicate = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![carry, eight])
            .unwrap()[0];
        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_scalar_operation_program() {
        // `f(x, y) = select(x > y, x + x, y)` staged through `ScalarOperation` tracers.
        let (output_type, program) = EagerContext::<Scalar, ScalarOperation<Scalar>>::trace(
            |(x, y)| {
                let mask = x.clone().greater_than(&y)?;
                Select::select(&mask, &(x.clone() + x), &y)
            },
            (DataType::F64, DataType::F64),
        )
        .unwrap();
        assert_eq!(output_type, DataType::F64);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:bool = compare [direction=GreaterThan] %0 %1
                    %3:f64 = add %0 %0
                    %4:f64 = select %2 %3 %1
                in (%4)
            "}
            .trim_end(),
        );

        // Interpreting the staged program exercises the in-band Boolean condition encoding of scalar values.
        assert_eq!(program.interpret((Scalar::from(3.0), Scalar::from(2.0))), Ok(Scalar::from(6.0)));
        assert_eq!(program.interpret((Scalar::from(1.0), Scalar::from(2.0))), Ok(Scalar::from(2.0)));
    }

    #[test]
    fn test_scalar_while() {
        let operation = WhileOperation::new();
        let condition = scalar_less_than_eight_condition();
        let body = scalar_doubling_body();
        let interfaces = [condition.interface(), body.interface()];

        // Operation identity, type inference, and eager binding through detached region access.
        assert_eq!(Operation::<DataType>::name(&operation), crate::operations::control_flow::WHILE_OPERATION_NAME);
        assert_eq!(operation.iteration_bound(), None);
        assert_eq!(operation.infer_output_types(&[DataType::F64], &interfaces), Ok(vec![DataType::F64]));
        assert_eq!(
            crate::EagerContext::<Scalar, ScalarOperation<Scalar>>::new().bind(
                ScalarOperation::While(operation),
                vec![condition.clone(), body.clone()],
                &[Scalar::from(1.0)],
            ),
            Ok(vec![Scalar::from(8.0)])
        );

        // Staging renders the nested condition and body programs, and interpretation runs the loop to completion.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (output_type, program) = EagerContext::<Scalar, ScalarOperation<Scalar>>::trace(
            |carry| {
                let operation = WhileOperation::new();
                let mut outputs = carry.context().stage_operation(
                    operation,
                    vec![scalar_less_than_eight_condition(), scalar_doubling_body()],
                    &[&carry],
                )?;
                Ok(outputs.remove(0))
            },
            DataType::F64,
        )
        .unwrap();
        assert_eq!(output_type, DataType::F64);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = while %0 [
                    condition={
                        lambda %0:f64 .
                        let %1:f64 = const
                            %2:bool = compare [direction=LessThan] %0 %1
                        in (%2)
                    },
                    body={
                        lambda %0:f64 .
                        let %1:f64 = add %0 %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(program.interpret(Scalar::from(1.0)), Ok(Scalar::from(8.0)));

        // The eager while JVP rule executes the loop directly over duals, so tangents flow through the doubling body.
        let (primal, tangent): (Scalar, Scalar) = domain
            .jvp(
                |carry| {
                    let operation = WhileOperation::new();
                    let regions = vec![scalar_less_than_eight_condition(), scalar_doubling_body()];
                    Ok(carry.context().bind(operation, regions, std::slice::from_ref(&carry))?.remove(0))
                },
                Scalar::from(1.0),
                Scalar::from(1.0),
            )
            .unwrap();
        assert_eq!(primal, 8.0);
        assert_eq!(tangent, 8.0);
    }

    #[test]
    fn test_scalar_while_rejects_non_boolean_condition() {
        let body = scalar_doubling_body();
        let interface = body.interface();
        assert_eq!(
            WhileOperation::new().infer_output_types(&[DataType::F64], &[interface.clone(), interface]),
            Err(TypeError { message: "'while' condition output type must be bool, but got f64".to_string() }),
        );
    }

    #[test]
    fn test_scalar_token() {
        let token = Scalar::Token;

        // The token scalar carries no payload: it renders as `token`, supports no arithmetic, comparisons, or
        // Boolean conversion, and its `zero_like`/`one_like` are the identity.
        assert_eq!(token.r#type().into_owned(), DataType::Token);
        assert_eq!(token.to_string(), "token");
        assert_eq!(token.as_boolean(), token);
        assert!(token.boolean().is_err());
        assert!(token.compare(&token, ComparisonDirection::Equal).is_err());
        assert!(Neg::neg(&token).is_err());
        assert!(Add::add(&token, &token).is_err());
        assert_eq!(token.zero_like(), token);
        assert_eq!(token.one_like(), token);

        // The token data type has no zero or one constant, and a same-type conversion is the identity.
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        assert!(context.zero(&DataType::Token).is_err());
        assert!(context.one(&DataType::Token).is_err());
        assert_eq!(token.convert_element_type(DataType::Token), Ok(token));
    }

    #[test]
    fn test_scalar_zero() {
        let zero = Scalar::Zero;

        // The zero-space scalar is the unique value of its type. Its total equality and ordering operations therefore
        // compare it equal only to itself, while semantic comparison, arithmetic, and predicate conversion remain
        // unsupported.
        assert_eq!(zero.r#type().into_owned(), DataType::Zero);
        assert_eq!(zero.to_string(), "zero");
        assert_eq!(zero, Scalar::Zero);
        assert_ne!(zero, Scalar::from(false));
        assert_eq!(zero.partial_cmp(&Scalar::Zero), Some(Ordering::Equal));
        assert_eq!(zero.as_boolean(), zero);
        assert!(zero.boolean().is_err());
        assert!(zero.compare(&zero, ComparisonDirection::Equal).is_err());
        assert!(Neg::neg(&zero).is_err());
        assert!(Add::add(&zero, &zero).is_err());
        assert_eq!(zero.zero_like(), zero);
        assert_eq!(zero.one_like(), zero);

        // Additive-zero construction and identity conversion produce the unique value. Multiplicative-one
        // construction and conversions to or from ordinary element types are rejected.
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        assert_eq!(context.zero(&DataType::Zero), Ok(zero));
        assert!(context.one(&DataType::Zero).is_err());
        assert_eq!(zero.convert_element_type(DataType::Zero), Ok(zero));
        assert!(zero.convert_element_type(DataType::Boolean).is_err());
        assert!(Scalar::from(false).convert_element_type(DataType::Zero).is_err());
        assert_eq!(zero.promote_element_type(DataType::Zero), Ok(zero));
        assert!(zero.promote_element_type(DataType::Boolean).is_err());
    }

    #[test]
    fn test_scalar_equality_and_ordering() {
        // Same-variant payloads compare by value, and scalars of different data types never compare equal.
        assert_eq!(Scalar::from(1.5f64), Scalar::from(1.5f64));
        assert_ne!(Scalar::from(1.5f64), Scalar::from(2.5f64));
        assert_ne!(Scalar::from(1.5f32), Scalar::from(1.5f64));
        assert_ne!(Scalar::from(1i32), Scalar::from(1i64));

        // Equality against bare primitives works in both directions and only within the matching variant.
        assert_eq!(Scalar::from(1.5f32), 1.5f32);
        assert_eq!(1.5f64, Scalar::from(1.5f64));
        assert_ne!(Scalar::from(1.5f32), 1.5f64);

        // Ordering is defined within a variant and undefined across variants (and for complex scalars, which the
        // complex tests cover).
        assert!(Scalar::from(1i32) < Scalar::from(2i32));
        assert!(Scalar::from(2.5f64) > Scalar::from(1.5f64));
        assert_eq!(Scalar::from(1i32).partial_cmp(&Scalar::from(1i64)), None);

        // Approximate equality compares exactly widened floating-point payloads, including across variants, and
        // against bare `f64` values.
        assert_abs_diff_eq!(Scalar::from(1.5f32), Scalar::from(1.5f64 + 1e-12), epsilon = 1e-9);
        assert_abs_diff_eq!(Scalar::from(2.0f32), 2.0f64, epsilon = 1e-9);
    }

    #[test]
    fn test_scalar_low_precision_float_types_and_bits() {
        // For every low-precision format, the listed bit pattern encodes the value one.
        let values = [
            (DataType::F4E2M1FN, 0x02),
            (DataType::F8E3M4, 0x30),
            (DataType::F8E4M3, 0x38),
            (DataType::F8E4M3FN, 0x38),
            (DataType::F8E4M3FNUZ, 0x40),
            (DataType::F8E4M3B11FNUZ, 0x58),
            (DataType::F8E5M2, 0x3c),
            (DataType::F8E5M2FNUZ, 0x40),
            (DataType::F8E8M0FNU, 0x7f),
        ];
        for (r#type, bits) in values {
            let scalar = Scalar::from_low_precision_float_bits(r#type, bits).unwrap();
            assert_eq!(scalar.r#type().into_owned(), r#type);
            assert_eq!(scalar.low_precision_float_bits(), Some(bits));
            assert_eq!(scalar.to_string(), "1");
            assert_eq!(scalar.boolean(), Ok(true));
            assert_eq!(scalar.one_like(), scalar);
            assert_eq!(scalar.convert_element_type(r#type), Ok(scalar));
        }

        // Construction validates the four-bit format and rejects non-low-precision data types, while equality
        // compares decoded values: signed zeros are equal and NaN payloads are unequal to themselves.
        assert!(Scalar::from_low_precision_float_bits(DataType::F4E2M1FN, 0x10).is_err());
        assert!(Scalar::from_low_precision_float_bits(DataType::F32, 0).is_err());
        assert_eq!(Scalar::F8E4M3(0), Scalar::F8E4M3(0x80));
        assert_ne!(Scalar::F8E4M3(0x79), Scalar::F8E4M3(0x79));
    }

    #[test]
    fn test_scalar_f8e8m0fnu_nan_encoding() {
        let nan = Scalar::from_low_precision_float_bits(DataType::F8E8M0FNU, 0xff).unwrap();

        // `0xff` is the format's sole NaN encoding, so it follows ordinary floating-point equality and ordering
        // semantics after exact widening.
        assert_ne!(nan, nan);
        assert_eq!(nan.partial_cmp(&nan), None);
        assert_eq!(nan.compare(&nan, ComparisonDirection::Equal), Ok(Scalar::Bool(false)));
        assert_eq!(nan.compare(&nan, ComparisonDirection::NotEqual), Ok(Scalar::Bool(true)));
        let Scalar::F64(widened) = nan.convert_element_type(DataType::F64).unwrap() else {
            panic!("expected an f64 scalar")
        };
        assert!(widened.is_nan());

        // Explicit narrowing canonicalizes any NaN payload to `0xff`; the format's existing zero and negation
        // restrictions remain unchanged.
        assert!(matches!(
            Scalar::from(f64::NAN).convert_element_type(DataType::F8E8M0FNU),
            Ok(Scalar::F8E8M0FNU(0xff)),
        ));
        assert!(Scalar::from(0.0).convert_element_type(DataType::F8E8M0FNU).is_err());
        assert!(Neg::neg(&Scalar::F8E8M0FNU(0x7f)).is_err());
    }

    #[test]
    fn test_scalar_low_precision_float_arithmetic() {
        let one = Scalar::F8E4M3(0x38);
        let two = Scalar::F8E4M3(0x40);
        let negative_one = Scalar::F8E4M3(0xb8);

        // Low-precision arithmetic computes through the decoded `f64` values and re-encodes the results.
        assert_eq!(Add::add(&one, &one), Ok(two));
        assert_eq!(Mul::mul(&two, &negative_one), Ok(Scalar::F8E4M3(0xc0)));
        assert_eq!(Neg::neg(&one), Ok(negative_one));
        assert_eq!(Abs::abs(&negative_one), Ok(one));
        assert_eq!(one.compare(&two, ComparisonDirection::LessThan), Ok(Scalar::Bool(true)));
        assert!(Add::add(&one, &Scalar::F8E5M2(0x3c)).is_err());
        assert!(Neg::neg(&Scalar::F8E8M0FNU(0x7f)).is_err());

        // The zero and one constants use the formats' canonical encodings, and the unsigned power-of-two format
        // `f8e8m0fnu` cannot represent zero.
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        assert_eq!(context.zero(&DataType::F8E4M3), Ok(Scalar::F8E4M3(0)));
        assert_eq!(context.one(&DataType::F8E4M3), Ok(one));
        assert!(context.zero(&DataType::F8E8M0FNU).is_err());
    }

    #[test]
    fn test_scalar_display() {
        // Numeric scalars render their payloads directly.
        assert_eq!(Scalar::from(true).to_string(), "true");
        assert_eq!(Scalar::from(-3i32).to_string(), "-3");
        assert_eq!(Scalar::from(1.5f64).to_string(), "1.5");

        // Complex scalars render as `<real>+<imaginary>i`, folding a negative imaginary part's sign into the
        // separator.
        assert_eq!(Scalar::from(Complex::new(1.5f32, 2.0f32)).to_string(), "1.5+2i");
        assert_eq!(Scalar::from(Complex::new(1.5f32, -2.0f32)).to_string(), "1.5-2i");
        assert_eq!(Scalar::from(Complex::new(-0.5f64, 0.25f64)).to_string(), "-0.5+0.25i");
        assert_eq!(Scalar::from(Complex::new(0.0f64, -1.0f64)).to_string(), "0-1i");
    }

    #[test]
    fn test_scalar_domain() {
        // `EagerContext<Scalar, ScalarOperation<Scalar>>` is a zero-sized token.
        assert_eq!(size_of::<EagerContext<Scalar, ScalarOperation<Scalar>>>(), 0);

        // It is an eager `Context`. Binding a nullary zero/one operation interprets it directly over concrete
        // `Scalar` values, yielding the corresponding scalar identity for the requested `DataType`.
        assert_eq!(
            EagerContext::<Scalar, ScalarOperation<Scalar>>::new().bind(
                ZeroOperation::new(DataType::F64),
                Vec::new(),
                &[]
            ),
            Ok(vec![Scalar::from(0.0)]),
        );
        assert_eq!(
            EagerContext::<Scalar, ScalarOperation<Scalar>>::default().bind(
                OneOperation::new(DataType::F64),
                Vec::new(),
                &[]
            ),
            Ok(vec![Scalar::from(1.0)]),
        );
    }

    #[test]
    fn test_scalar_boolean_like() {
        // Boolean conversion treats any nonzero payload as true, decoding Booleans, integers, and floating-point
        // scalars alike, and `as_boolean` re-wraps that truth value as a `Scalar::Bool`.
        assert_eq!(Scalar::from(true).boolean(), Ok(true));
        assert_eq!(Scalar::from(0i32).boolean(), Ok(false));
        assert_eq!(Scalar::from(-2i64).boolean(), Ok(true));
        assert_eq!(Scalar::from(0.0f64).boolean(), Ok(false));
        assert_eq!(Scalar::from(0.5f32).boolean(), Ok(true));
        assert_eq!(Scalar::from(2.5f64).as_boolean(), Scalar::from(true));
    }

    #[test]
    fn test_scalar_constants() {
        // The zero and one constants exist for every numeric data type.
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        assert_eq!(context.zero(&DataType::I32), Ok(Scalar::from(0i32)));
        assert_eq!(context.one(&DataType::Boolean), Ok(Scalar::from(true)));
        assert_eq!(context.zero(&DataType::F32), Ok(Scalar::from(0.0f32)));

        // The like-typed constants adopt the source scalar's variant.
        assert_eq!(Scalar::from(5i16).zero_like(), Scalar::from(0i16));
        assert_eq!(Scalar::from(2.5f64).one_like(), Scalar::from(1.0f64));
    }

    #[test]
    fn test_scalar_arithmetic() {
        // Arithmetic computes in the promoted payload type, through both the fallible capabilities and the `std::ops`
        // operator sugar layered on top of them.
        assert_eq!(Add::add(&Scalar::from(2i32), &Scalar::from(3i32)), Ok(Scalar::from(5i32)));
        assert_eq!(Add::add(&Scalar::from(1.5f32), &Scalar::from(1.5f64)), Ok(Scalar::from(3.0f64)));
        assert_eq!(Add::add(&Scalar::from(-1i8), &Scalar::from(2u8)), Ok(Scalar::from(1i16)));
        assert_eq!(Scalar::from(2.0) + Scalar::from(3.0), Scalar::from(5.0));
        assert_eq!(Scalar::from(2.0) - Scalar::from(3.0), Scalar::from(-1.0));
        assert_eq!(Scalar::from(2.0) * Scalar::from(3.0), Scalar::from(6.0));
        assert_eq!(Scalar::from(7i64) / Scalar::from(2i64), Scalar::from(3i64));
        assert_eq!(-Scalar::from(2.0), Scalar::from(-2.0));

        // Fixed-width integer negation, addition, subtraction, and multiplication wrap deterministically.
        assert_eq!(Neg::neg(&Scalar::from(i8::MIN)), Ok(Scalar::from(i8::MIN)));
        assert_eq!(Neg::neg(&Scalar::from(1u8)), Ok(Scalar::from(u8::MAX)));
        assert_eq!(Add::add(&Scalar::from(i8::MAX), &Scalar::from(1i8)), Ok(Scalar::from(i8::MIN)));
        assert_eq!(Sub::sub(&Scalar::from(i8::MIN), &Scalar::from(1i8)), Ok(Scalar::from(i8::MAX)));
        assert_eq!(Mul::mul(&Scalar::from(i8::MAX), &Scalar::from(2i8)), Ok(Scalar::from(-2i8)));

        // Fallible integer division reports exceptional inputs instead of panicking.
        assert_eq!(
            Div::div(&Scalar::from(1i32), &Scalar::from(0i32)),
            Err(ProgramError::Type(TypeError {
                message: "cannot divide an integer scalar of data type i32 by zero".to_string(),
            })),
        );
        assert_eq!(
            Div::div(&Scalar::from(i32::MIN), &Scalar::from(-1i32)),
            Err(ProgramError::Type(TypeError {
                message: "cannot divide the minimum integer scalar of data type i32 by -1".to_string(),
            })),
        );

        // Non-numeric scalars remain outside the arithmetic operation family.
        assert_eq!(
            Add::add(&Scalar::from(true), &Scalar::from(true)),
            Err(ProgramError::Type(TypeError { message: "'add' does not support input data type bool".to_string() })),
        );
    }

    #[test]
    fn test_scalar_logical_operations() {
        // Boolean scalars combine logically and integer scalars combine bitwise, through both the fallible
        // capabilities and the `std::ops` operator sugar layered on top of them.
        assert_eq!(Scalar::from(true) & Scalar::from(false), Scalar::from(false));
        assert_eq!(Scalar::from(true) | Scalar::from(false), Scalar::from(true));
        assert_eq!(Scalar::from(true) ^ Scalar::from(true), Scalar::from(false));
        assert_eq!(!Scalar::from(true), Scalar::from(false));
        assert_eq!(Scalar::from(0b1100_u8) & Scalar::from(0b1010_u8), Scalar::from(0b1000_u8));
        assert_eq!(Scalar::from(0b1100_u8) | Scalar::from(0b1010_u8), Scalar::from(0b1110_u8));
        assert_eq!(Scalar::from(0b1100_u8) ^ Scalar::from(0b1010_u8), Scalar::from(0b0110_u8));
        assert_eq!(!Scalar::from(0b1100_u8), Scalar::from(!0b1100_u8));

        // Mismatched and unsupported data types surface `TypeError`s through the fallible capabilities.
        assert!(And::and(&Scalar::from(true), &Scalar::from(1.0)).is_err());
        assert!(Or::or(&Scalar::from(1.0), &Scalar::from(2.0)).is_err());
        assert!(Xor::xor(&Scalar::from(0b1100_u8), &Scalar::from(0b1010_u16)).is_err());
        assert!(Not::not(&Scalar::from(1.0)).is_err());

        // `f(x, y) = not((x > y or x < y) xor (x > y and x < y))`, which reduces to `x == y`, staged through
        // `ScalarOperation` tracers so that all four logical operations appear in one scalar program.
        let (output_type, program) = EagerContext::<Scalar, ScalarOperation<Scalar>>::trace(
            |(x, y)| {
                let greater = x.clone().greater_than(&y)?;
                let less = x.less_than(&y)?;
                Ok(!((greater.clone() | less.clone()) ^ (greater & less)))
            },
            (DataType::F64, DataType::F64),
        )
        .unwrap();
        assert_eq!(output_type, DataType::Boolean);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:bool = compare [direction=GreaterThan] %0 %1
                    %3:bool = compare [direction=LessThan] %0 %1
                    %4:bool = or %2 %3
                    %5:bool = and %2 %3
                    %6:bool = xor %4 %5
                    %7:bool = not %6
                in (%7)
            "}
            .trim_end(),
        );

        // Interpreting the staged program replays the logical operations on concrete scalars.
        assert_eq!(program.interpret((Scalar::from(3.0), Scalar::from(2.0))), Ok(Scalar::from(false)));
        assert_eq!(program.interpret((Scalar::from(2.0), Scalar::from(2.0))), Ok(Scalar::from(true)));
    }

    #[test]
    fn test_scalar_math() {
        // The elementwise math capabilities compute in the payload type for the floating-point variants.
        assert_eq!(Scalar::from(0.0).sin(), Ok(Scalar::from(0.0)));
        assert_eq!(Scalar::from(0.0).cos(), Ok(Scalar::from(1.0)));
        assert_eq!(Scalar::from(0.0).exp(), Ok(Scalar::from(1.0)));
        assert_eq!(Scalar::from(1.0).log(), Ok(Scalar::from(0.0)));
        assert_eq!(Scalar::from(4.0).sqrt(), Ok(Scalar::from(2.0)));
        assert_abs_diff_eq!(
            Scalar::from(1.0).atan2(&Scalar::from(1.0)).unwrap(),
            std::f64::consts::FRAC_PI_4,
            epsilon = 1e-12,
        );

        // The absolute value covers signed integers, floating-point values, and complex magnitudes (with a real
        // result).
        assert_eq!(Scalar::from(-2i32).abs(), Ok(Scalar::from(2i32)));
        assert_eq!(Scalar::from(i8::MIN).abs(), Ok(Scalar::from(i8::MIN)));
        assert_eq!(Scalar::from(i16::MIN).abs(), Ok(Scalar::from(i16::MIN)));
        assert_eq!(Scalar::from(i32::MIN).abs(), Ok(Scalar::from(i32::MIN)));
        assert_eq!(Scalar::from(i64::MIN).abs(), Ok(Scalar::from(i64::MIN)));
        assert_eq!(Scalar::from(-2.5f64).abs(), Ok(Scalar::from(2.5f64)));
        assert_eq!(Scalar::from(Complex::new(3.0f32, 4.0f32)).abs(), Ok(Scalar::from(5.0f32)));

        // Unsupported data types surface `TypeError`s.
        assert!(Scalar::from(true).sin().is_err());
        assert!(Scalar::from(1i32).exp().is_err());
        assert!(Scalar::from(true).abs().is_err());
        assert_eq!(Scalar::from(1.0f32).atan2(&Scalar::from(1.0f64)), Ok(Scalar::from(std::f64::consts::FRAC_PI_4)),);
    }

    #[test]
    fn test_scalar_complex_parts() {
        // Complex construction pairs same-precision real parts, and the part extractions invert it.
        let complex = crate::operations::complex::Complex::complex(&Scalar::from(1.5f32), &Scalar::from(-2.0f32));
        assert_eq!(complex, Ok(Scalar::from(Complex::new(1.5f32, -2.0f32))));
        let complex = complex.unwrap();
        assert_eq!(complex.conjugate(), Ok(Scalar::from(Complex::new(1.5f32, 2.0f32))));
        assert_eq!(complex.real(), Ok(Scalar::from(1.5f32)));
        assert_eq!(complex.imaginary(), Ok(Scalar::from(-2.0f32)));

        // Mixed-precision construction and part extraction from non-complex scalars surface `TypeError`s.
        assert!(crate::operations::complex::Complex::complex(&Scalar::from(1.5f32), &Scalar::from(1.5f64)).is_err());
        assert!(Scalar::from(1.5f64).conjugate().is_err());
        assert!(Scalar::from(1.5f64).real().is_err());
        assert!(Scalar::from(1.5f64).imaginary().is_err());
    }

    #[test]
    fn test_scalar_complex_arithmetic() {
        let left = Scalar::from(Complex::new(1.0f64, 2.0f64));
        let right = Scalar::from(Complex::new(3.0f64, -1.0f64));

        // The complex variants support the full fallible arithmetic surface, computing in the complex field.
        assert_eq!(left + right, Scalar::from(Complex::new(4.0f64, 1.0f64)));
        assert_eq!(left - right, Scalar::from(Complex::new(-2.0f64, 3.0f64)));
        assert_eq!(left * right, Scalar::from(Complex::new(5.0f64, 5.0f64)));
        assert_eq!(-left, Scalar::from(Complex::new(-1.0f64, -2.0f64)));
        assert_abs_diff_eq!((left / right) * right, left, epsilon = 1e-15);
        assert_eq!(
            Div::div(&Scalar::from(Complex::new(1.0e308f64, 0.0)), &Scalar::from(Complex::new(1.0e308f64, 0.0)),),
            Ok(Scalar::from(Complex::new(1.0f64, 0.0))),
        );

        // Complex sine and cosine implement the analytic continuations. Their cancellation-resistant hyperbolic
        // decomposition can round its final component differently from `num_complex` by one ulp.
        assert_abs_diff_eq!(left.sin().unwrap(), Scalar::from(Complex::new(1.0f64, 2.0f64).sin()), epsilon = 1e-15,);
        assert_abs_diff_eq!(left.cos().unwrap(), Scalar::from(Complex::new(1.0f64, 2.0f64).cos()), epsilon = 1e-15,);

        // Constants and Boolean-ness: zero/one carry a zero imaginary part, and a complex scalar is truthy exactly
        // when it is not the complex zero.
        assert_eq!(left.zero_like(), Scalar::from(Complex::new(0.0f64, 0.0f64)));
        assert_eq!(left.one_like(), Scalar::from(Complex::new(1.0f64, 0.0f64)));
        assert_eq!(
            EagerContext::<Scalar, ScalarOperation<Scalar>>::new().bind(
                ZeroOperation::new(DataType::C64),
                Vec::new(),
                &[]
            ),
            Ok(vec![Scalar::from(Complex::new(0.0f32, 0.0f32))]),
        );
        assert_eq!(left.boolean(), Ok(true));
        assert_eq!(left.zero_like().boolean(), Ok(false));
        assert_eq!(Scalar::from(Complex::new(0.0f64, 0.5f64)).boolean(), Ok(true));

        // Complex scalars are unordered.
        assert_eq!(left.partial_cmp(&right), None);

        // Mixed real and complex arithmetic promotes the real operand into the complex field.
        assert_eq!(Add::add(&left, &Scalar::from(1.0f64)), Ok(Scalar::from(Complex::new(2.0f64, 2.0f64))),);
    }

    #[test]
    fn test_scalar_complex_program_constant_rendering() {
        // A complex constant is staged and rendered like any other constant (a `const` binding typed `c64`; the
        // value-literal syntax itself is covered by the `Display` test above), and interpretation recovers the
        // carried complex value.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::C64);
        let constant = builder.add_constant(Scalar::from(Complex::new(1.5f32, -2.0f32)));
        let output = builder.add_instruction(MulOperation, Vec::new(), vec![input, constant]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:c64 .
                let %1:c64 = const
                    %2:c64 = mul %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Interpreting the program multiplies in the complex field.
        let outputs = program.interpret(vec![Scalar::from(Complex::new(2.0f32, 1.0f32))]).unwrap();
        assert_eq!(outputs, vec![Scalar::from(Complex::new(2.0f32, 1.0f32) * Complex::new(1.5f32, -2.0f32))]);
    }

    #[test]
    fn test_scalar_compare() {
        // Comparisons return honestly Boolean-typed scalars for every ordered direction.
        let one = Scalar::from(1.0);
        let two = Scalar::from(2.0);
        assert_eq!(one.compare(&two, ComparisonDirection::LessThan), Ok(Scalar::Bool(true)));
        assert_eq!(one.compare(&two, ComparisonDirection::LessThanOrEqual), Ok(Scalar::Bool(true)));
        assert_eq!(one.compare(&two, ComparisonDirection::GreaterThan), Ok(Scalar::Bool(false)));
        assert_eq!(one.compare(&one, ComparisonDirection::GreaterThanOrEqual), Ok(Scalar::Bool(true)));
        assert_eq!(one.compare(&two, ComparisonDirection::Equal), Ok(Scalar::Bool(false)));
        assert_eq!(one.compare(&two, ComparisonDirection::NotEqual), Ok(Scalar::Bool(true)));

        // An unordered pair (one involving a NaN) satisfies only the `NotEqual` direction.
        let nan = Scalar::from(f64::NAN);
        assert_eq!(one.compare(&nan, ComparisonDirection::LessThan), Ok(Scalar::Bool(false)));
        assert_eq!(one.compare(&nan, ComparisonDirection::NotEqual), Ok(Scalar::Bool(true)));

        // Mismatched variants are rejected.
        assert!(Scalar::from(1.0f32).compare(&Scalar::from(1.0f64), ComparisonDirection::Equal).is_err());

        // Only the equality directions are defined for the unordered complex scalars.
        let left = Scalar::from(Complex::new(1.0f64, 2.0f64));
        let right = Scalar::from(Complex::new(3.0f64, -1.0f64));
        assert_eq!(left.compare(&left, ComparisonDirection::Equal), Ok(Scalar::Bool(true)));
        assert_eq!(left.compare(&right, ComparisonDirection::Equal), Ok(Scalar::Bool(false)));
        assert_eq!(left.compare(&right, ComparisonDirection::NotEqual), Ok(Scalar::Bool(true)));
        assert_eq!(
            left.compare(&right, ComparisonDirection::LessThan),
            Err(TypeError {
                message: "cannot apply an ordered comparison to unordered complex scalars of data type c128"
                    .to_string(),
            }
            .into()),
        );
    }

    #[test]
    fn test_scalar_convert_element_type() {
        // Same-type conversions are the identity, while widening conversions preserve values across floating-point,
        // integer, Boolean, and complex representations.
        assert_eq!(Scalar::from(2.5f32).convert_element_type(DataType::F32), Ok(Scalar::from(2.5f32)));
        assert_eq!(Scalar::from(2.5f32).convert_element_type(DataType::F64), Ok(Scalar::from(2.5f64)));
        assert_eq!(Scalar::from(3i16).convert_element_type(DataType::I32), Ok(Scalar::from(3i32)));
        assert_eq!(Scalar::from(true).convert_element_type(DataType::U16), Ok(Scalar::from(1u16)));
        assert_eq!(
            Scalar::from(Complex::new(1.5f32, -2.0f32)).convert_element_type(DataType::C128),
            Ok(Scalar::from(Complex::new(1.5f64, -2.0f64))),
        );

        // Explicit conversions also support narrowing and complex-to-real conversion.
        assert_eq!(Scalar::from(2.5f64).convert_element_type(DataType::F32), Ok(Scalar::from(2.5f32)));
        assert_eq!(
            Scalar::from(Complex::new(1.5f64, -2.0f64)).convert_element_type(DataType::F32),
            Ok(Scalar::from(1.5f32)),
        );
        assert_eq!(
            Scalar::from(Complex::new(0.0f64, -2.0f64)).convert_element_type(DataType::Boolean),
            Ok(Scalar::from(true)),
        );
    }

    #[test]
    fn test_scalar_promote_element_type() {
        assert_eq!(Scalar::from(2.5f32).promote_element_type(DataType::F64), Ok(Scalar::from(2.5f64)));
        assert_eq!(
            Scalar::from(2.5f64).promote_element_type(DataType::I32),
            Err(TypeError { message: "cannot promote type `f64` to type `i32`".to_string() }.into()),
        );
        assert_eq!(
            Scalar::from(Complex::new(1.5f32, -2.0f32)).promote_element_type(DataType::F64),
            Err(TypeError { message: "cannot promote type `c64` to type `f64`".to_string() }.into()),
        );
    }

    #[test]
    fn test_scalar_select() {
        // Selection promotes the selected branch to the promotion of the two branch data types, like `jnp.where`.
        assert_eq!(
            Select::select(&Scalar::from(true), &Scalar::from(1.5f32), &Scalar::from(2.5f64)),
            Ok(Scalar::from(1.5f64)),
        );
        assert_eq!(
            Select::select(&Scalar::from(false), &Scalar::from(1.5f32), &Scalar::from(2.5f64)),
            Ok(Scalar::from(2.5f64)),
        );

        // Scalar selection decodes the condition's in-band Boolean payload.
        assert_eq!(Select::select(&Scalar::from(1.0), &Scalar::from(1), &Scalar::from(2)), Ok(Scalar::from(1)));
        assert_eq!(Select::select(&Scalar::from(0.0), &Scalar::from(1), &Scalar::from(2)), Ok(Scalar::from(2)));
        assert!(Select::select(&Scalar::Token, &Scalar::from(1), &Scalar::from(2)).is_err());
    }
}
