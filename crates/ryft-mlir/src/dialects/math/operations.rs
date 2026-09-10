use crate::dialects::arith::attributes::{
    FastMathFlags, FastMathFlagsAttributeRef, RoundingMode, RoundingModeAttributeRef,
};
use crate::macros::{mlir_op, mlir_op_trait};
use crate::{
    Attribute, DetachedOp, DialectHandle, Error, FloatTypeRef, IndexTypeRef, IntegerTypeRef, Location, Operation,
    OperationBuilder, OperationResultRef, ShapedType, TensorTypeRef, Type, TypeRef, UnrankedTensorTypeRef, Value,
    ValueRef, VectorTypeDimension, VectorTypeRef,
};

/// Optional attributes shared by Math operation constructors.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MathOperationOptions {
    /// Fast-math assumptions for floating-point operations. Integer operations require [`FastMathFlags::NONE`].
    pub fastmath: FastMathFlags,
}

impl Default for MathOperationOptions {
    fn default() -> Self {
        Self { fastmath: FastMathFlags::NONE }
    }
}

/// Optional attributes for [`FmaOperation`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FmaOptions {
    /// Fast-math assumptions.
    pub fastmath: FastMathFlags,

    /// Explicit IEEE-754 rounding mode. `None` leaves the operation default unchanged.
    pub rounding_mode: Option<RoundingMode>,
}

impl Default for FmaOptions {
    fn default() -> Self {
        Self { fastmath: FastMathFlags::NONE, rounding_mode: None }
    }
}

/// Name of the fast-math flags attribute on floating-point Math operations.
pub const FASTMATH_ATTRIBUTE: &str = "fastmath";

/// Math [`Operation`] that computes a floating-point absolute value. Applies element-wise to scalar, vector, or tensor
/// operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`AbsFOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.absf %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathabsf-mathabsfop
pub trait AbsFOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.absf`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(AbsF);
mlir_op_trait!(AbsF, OneOperand);
mlir_op_trait!(AbsF, OneResult);
mlir_op_trait!(AbsF, ZeroRegions);
mlir_op_trait!(AbsF, ZeroSuccessors);

/// Constructs a new detached/owned [`AbsFOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`AbsFOperation`] for more information on the operation semantics.
pub fn absf<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedAbsFOperation<'c, 't>, Error> {
    absf_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`AbsFOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`AbsFOperation`] for more information on the operation semantics.
pub fn absf_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedAbsFOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.absf")?;
    let mut builder = OperationBuilder::new("math.absf", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::absf`"))
    })
}
/// Math [`Operation`] that computes an integer absolute value. Applies element-wise to scalar, vector, or tensor
/// operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`AbsIOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.absi %input : i32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathabsi-mathabsiop
pub trait AbsIOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(AbsI);
mlir_op_trait!(AbsI, OneOperand);
mlir_op_trait!(AbsI, OneResult);
mlir_op_trait!(AbsI, ZeroRegions);
mlir_op_trait!(AbsI, ZeroSuccessors);

/// Constructs a new detached/owned [`AbsIOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`AbsIOperation`] for more information on the operation semantics.
pub fn absi<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedAbsIOperation<'c, 't>, Error> {
    absi_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`AbsIOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`AbsIOperation`] for more information on the operation semantics.
pub fn absi_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedAbsIOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_integer_like(result_type, "math.absi")?;
    let builder = OperationBuilder::new("math.absi", location);
    if options.fastmath != FastMathFlags::NONE {
        return Err(Error::invalid_argument("fast-math flags are unsupported for integer Math operations"));
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::absi`"))
    })
}
/// Math [`Operation`] that computes an inverse hyperbolic cosine. Applies element-wise to scalar, vector, or tensor
/// operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`AcoshOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.acosh %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathacosh-mathacoshop
pub trait AcoshOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.acosh`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Acosh);
mlir_op_trait!(Acosh, OneOperand);
mlir_op_trait!(Acosh, OneResult);
mlir_op_trait!(Acosh, ZeroRegions);
mlir_op_trait!(Acosh, ZeroSuccessors);

/// Constructs a new detached/owned [`AcoshOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`AcoshOperation`] for more information on the operation semantics.
pub fn acosh<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedAcoshOperation<'c, 't>, Error> {
    acosh_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`AcoshOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`AcoshOperation`] for more information on the operation semantics.
pub fn acosh_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedAcoshOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.acosh")?;
    let mut builder = OperationBuilder::new("math.acosh", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::acosh`"))
    })
}
/// Math [`Operation`] that computes an inverse sine. Applies element-wise to scalar, vector, or tensor operands. The
/// result has the operand type.
///
/// # Example
///
/// The following is an example of a [`AsinOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.asin %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathasin-mathasinop
pub trait AsinOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.asin`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Asin);
mlir_op_trait!(Asin, OneOperand);
mlir_op_trait!(Asin, OneResult);
mlir_op_trait!(Asin, ZeroRegions);
mlir_op_trait!(Asin, ZeroSuccessors);

/// Constructs a new detached/owned [`AsinOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`AsinOperation`] for more information on the operation semantics.
pub fn asin<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedAsinOperation<'c, 't>, Error> {
    asin_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`AsinOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`AsinOperation`] for more information on the operation semantics.
pub fn asin_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedAsinOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.asin")?;
    let mut builder = OperationBuilder::new("math.asin", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::asin`"))
    })
}
/// Math [`Operation`] that computes an inverse hyperbolic sine. Applies element-wise to scalar, vector, or tensor
/// operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`AsinhOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.asinh %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathasinh-mathasinhop
pub trait AsinhOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.asinh`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Asinh);
mlir_op_trait!(Asinh, OneOperand);
mlir_op_trait!(Asinh, OneResult);
mlir_op_trait!(Asinh, ZeroRegions);
mlir_op_trait!(Asinh, ZeroSuccessors);

/// Constructs a new detached/owned [`AsinhOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`AsinhOperation`] for more information on the operation semantics.
pub fn asinh<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedAsinhOperation<'c, 't>, Error> {
    asinh_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`AsinhOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`AsinhOperation`] for more information on the operation semantics.
pub fn asinh_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedAsinhOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.asinh")?;
    let mut builder = OperationBuilder::new("math.asinh", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::asinh`"))
    })
}
/// Math [`Operation`] that computes an inverse tangent. Applies element-wise to scalar, vector, or tensor operands. The
/// result has the operand type.
///
/// # Example
///
/// The following is an example of a [`AtanOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.atan %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathatan-mathatanop
pub trait AtanOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.atan`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Atan);
mlir_op_trait!(Atan, OneOperand);
mlir_op_trait!(Atan, OneResult);
mlir_op_trait!(Atan, ZeroRegions);
mlir_op_trait!(Atan, ZeroSuccessors);

/// Constructs a new detached/owned [`AtanOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`AtanOperation`] for more information on the operation semantics.
pub fn atan<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedAtanOperation<'c, 't>, Error> {
    atan_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`AtanOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`AtanOperation`] for more information on the operation semantics.
pub fn atan_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedAtanOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.atan")?;
    let mut builder = OperationBuilder::new("math.atan", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::atan`"))
    })
}
/// Math [`Operation`] that computes an inverse hyperbolic tangent. Applies element-wise to scalar, vector, or tensor
/// operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`AtanhOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.atanh %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathatanh-mathatanhop
pub trait AtanhOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.atanh`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Atanh);
mlir_op_trait!(Atanh, OneOperand);
mlir_op_trait!(Atanh, OneResult);
mlir_op_trait!(Atanh, ZeroRegions);
mlir_op_trait!(Atanh, ZeroSuccessors);

/// Constructs a new detached/owned [`AtanhOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`AtanhOperation`] for more information on the operation semantics.
pub fn atanh<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedAtanhOperation<'c, 't>, Error> {
    atanh_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`AtanhOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`AtanhOperation`] for more information on the operation semantics.
pub fn atanh_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedAtanhOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.atanh")?;
    let mut builder = OperationBuilder::new("math.atanh", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::atanh`"))
    })
}
/// Math [`Operation`] that computes a two-argument inverse tangent. Applies element-wise to scalar, vector, or tensor
/// operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`Atan2Operation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.atan2 %lhs, %rhs : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathatan2-mathatan2op
pub trait Atan2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand-side input value.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right-hand-side input value.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.atan2`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Atan2);
mlir_op_trait!(Atan2, OneResult);
mlir_op_trait!(Atan2, ZeroRegions);
mlir_op_trait!(Atan2, ZeroSuccessors);

/// Constructs a new detached/owned [`Atan2Operation`] at the specified [`Location`]. Refer to the documentation of
/// [`Atan2Operation`] for more information on the operation semantics.
pub fn atan2<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    location: L,
) -> Result<DetachedAtan2Operation<'c, 't>, Error> {
    atan2_with_options(lhs, rhs, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`Atan2Operation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`Atan2Operation`] for more information on the operation semantics.
pub fn atan2_with_options<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedAtan2Operation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = lhs.r#type()?;
    if rhs.r#type()? != result_type {
        return Err(Error::invalid_argument("mismatched operand types for `math.atan2`"));
    }
    validate_float_like(result_type, "math.atan2")?;
    let mut builder = OperationBuilder::new("math.atan2", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder
        .add_operand(lhs)?
        .add_operand(rhs)?
        .add_result(result_type)?
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::atan2`"))
        })
}
/// Math [`Operation`] that computes a cube root. Applies element-wise to scalar, vector, or tensor operands. The result
/// has the operand type.
///
/// # Example
///
/// The following is an example of a [`CbrtOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.cbrt %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathcbrt-mathcbrtop
pub trait CbrtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.cbrt`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Cbrt);
mlir_op_trait!(Cbrt, OneOperand);
mlir_op_trait!(Cbrt, OneResult);
mlir_op_trait!(Cbrt, ZeroRegions);
mlir_op_trait!(Cbrt, ZeroSuccessors);

/// Constructs a new detached/owned [`CbrtOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`CbrtOperation`] for more information on the operation semantics.
pub fn cbrt<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedCbrtOperation<'c, 't>, Error> {
    cbrt_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`CbrtOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`CbrtOperation`] for more information on the operation semantics.
pub fn cbrt_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedCbrtOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.cbrt")?;
    let mut builder = OperationBuilder::new("math.cbrt", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::cbrt`"))
    })
}
/// Math [`Operation`] that computes a ceiling. The result is the least integral value greater than or equal to the
/// input. Applies element-wise to scalar, vector, or tensor operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`CeilOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.ceil %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathceil-mathceilop
pub trait CeilOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.ceil`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Ceil);
mlir_op_trait!(Ceil, OneOperand);
mlir_op_trait!(Ceil, OneResult);
mlir_op_trait!(Ceil, ZeroRegions);
mlir_op_trait!(Ceil, ZeroSuccessors);

/// Constructs a new detached/owned [`CeilOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`CeilOperation`] for more information on the operation semantics.
pub fn ceil<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedCeilOperation<'c, 't>, Error> {
    ceil_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`CeilOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`CeilOperation`] for more information on the operation semantics.
pub fn ceil_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedCeilOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.ceil")?;
    let mut builder = OperationBuilder::new("math.ceil", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::ceil`"))
    })
}
/// Math [`Operation`] that clamps a floating-point value. The first operand is bounded below by the second and above by
/// the third. Applies element-wise to scalar, vector, or tensor operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`ClampFOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.clampf %value to [%lower, %upper] : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathclampf-mathclampfop
pub trait ClampFOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the first input value.
    fn first(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the second input value.
    fn second(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the third input value.
    fn third(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.clampf`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(ClampF);
mlir_op_trait!(ClampF, OneResult);
mlir_op_trait!(ClampF, ZeroRegions);
mlir_op_trait!(ClampF, ZeroSuccessors);

/// Constructs a new detached/owned [`ClampFOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`ClampFOperation`] for more information on the operation semantics.
pub fn clampf<
    'first,
    'second,
    'third,
    'c: 'first + 'second + 'third,
    't: 'c,
    First: Value<'first, 'c, 't>,
    Second: Value<'second, 'c, 't>,
    Third: Value<'third, 'c, 't>,
    L: Location<'c, 't>,
>(
    first: First,
    second: Second,
    third: Third,
    location: L,
) -> Result<DetachedClampFOperation<'c, 't>, Error> {
    clampf_with_options(first, second, third, Default::default(), location)
}

/// Constructs a new detached/owned [`ClampFOperation`] with explicit options at the specified [`Location`]. Refer to
/// the documentation of [`ClampFOperation`] for more information on the operation semantics.
pub fn clampf_with_options<
    'first,
    'second,
    'third,
    'c: 'first + 'second + 'third,
    't: 'c,
    First: Value<'first, 'c, 't>,
    Second: Value<'second, 'c, 't>,
    Third: Value<'third, 'c, 't>,
    L: Location<'c, 't>,
>(
    first: First,
    second: Second,
    third: Third,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedClampFOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = first.r#type()?;
    if second.r#type()? != result_type || third.r#type()? != result_type {
        return Err(Error::invalid_argument("mismatched operand types for `math.clampf`"));
    }
    validate_float_like(result_type, "math.clampf")?;
    let mut builder = OperationBuilder::new("math.clampf", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder
        .add_operand(first)?
        .add_operand(second)?
        .add_operand(third)?
        .add_result(result_type)?
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::clampf`"))
        })
}
/// Math [`Operation`] that copies a floating-point sign. The result has the magnitude of `lhs` and the sign of `rhs`.
/// Applies element-wise to scalar, vector, or tensor operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`CopySignOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %result = math.copysign %lhs, %rhs : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathcopysign-mathcopysignop
pub trait CopySignOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand-side input value.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right-hand-side input value.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.copysign`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(CopySign);
mlir_op_trait!(CopySign, OneResult);
mlir_op_trait!(CopySign, ZeroRegions);
mlir_op_trait!(CopySign, ZeroSuccessors);

/// Constructs a new detached/owned [`CopySignOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`CopySignOperation`] for more information on the operation semantics.
pub fn copysign<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    location: L,
) -> Result<DetachedCopySignOperation<'c, 't>, Error> {
    copysign_with_options(lhs, rhs, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`CopySignOperation`] with explicit options at the specified [`Location`]. Refer to
/// the documentation of [`CopySignOperation`] for more information on the operation semantics.
pub fn copysign_with_options<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedCopySignOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = lhs.r#type()?;
    if rhs.r#type()? != result_type {
        return Err(Error::invalid_argument("mismatched operand types for `math.copysign`"));
    }
    validate_float_like(result_type, "math.copysign")?;
    let mut builder = OperationBuilder::new("math.copysign", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder
        .add_operand(lhs)?
        .add_operand(rhs)?
        .add_result(result_type)?
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::copysign`"))
        })
}
/// Math [`Operation`] that computes a cosine. Applies element-wise to scalar, vector, or tensor operands. The result
/// has the operand type.
///
/// # Example
///
/// The following is an example of a [`CosOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.cos %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathcos-mathcosop
pub trait CosOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.cos`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Cos);
mlir_op_trait!(Cos, OneOperand);
mlir_op_trait!(Cos, OneResult);
mlir_op_trait!(Cos, ZeroRegions);
mlir_op_trait!(Cos, ZeroSuccessors);

/// Constructs a new detached/owned [`CosOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`CosOperation`] for more information on the operation semantics.
pub fn cos<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedCosOperation<'c, 't>, Error> {
    cos_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`CosOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`CosOperation`] for more information on the operation semantics.
pub fn cos_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedCosOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.cos")?;
    let mut builder = OperationBuilder::new("math.cos", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::cos`"))
    })
}
/// Math [`Operation`] that computes an inverse cosine. Applies element-wise to scalar, vector, or tensor operands. The
/// result has the operand type.
///
/// # Example
///
/// The following is an example of a [`AcosOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.acos %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathacos-mathacosop
pub trait AcosOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.acos`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Acos);
mlir_op_trait!(Acos, OneOperand);
mlir_op_trait!(Acos, OneResult);
mlir_op_trait!(Acos, ZeroRegions);
mlir_op_trait!(Acos, ZeroSuccessors);

/// Constructs a new detached/owned [`AcosOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`AcosOperation`] for more information on the operation semantics.
pub fn acos<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedAcosOperation<'c, 't>, Error> {
    acos_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`AcosOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`AcosOperation`] for more information on the operation semantics.
pub fn acos_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedAcosOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.acos")?;
    let mut builder = OperationBuilder::new("math.acos", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::acos`"))
    })
}
/// Math [`Operation`] that computes a hyperbolic cosine. Applies element-wise to scalar, vector, or tensor operands.
/// The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`CoshOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.cosh %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathcosh-mathcoshop
pub trait CoshOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.cosh`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Cosh);
mlir_op_trait!(Cosh, OneOperand);
mlir_op_trait!(Cosh, OneResult);
mlir_op_trait!(Cosh, ZeroRegions);
mlir_op_trait!(Cosh, ZeroSuccessors);

/// Constructs a new detached/owned [`CoshOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`CoshOperation`] for more information on the operation semantics.
pub fn cosh<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedCoshOperation<'c, 't>, Error> {
    cosh_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`CoshOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`CoshOperation`] for more information on the operation semantics.
pub fn cosh_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedCoshOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.cosh")?;
    let mut builder = OperationBuilder::new("math.cosh", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::cosh`"))
    })
}
/// Math [`Operation`] that computes a sine. Applies element-wise to scalar, vector, or tensor operands. The result has
/// the operand type.
///
/// # Example
///
/// The following is an example of a [`SinOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.sin %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathsin-mathsinop
pub trait SinOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.sin`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Sin);
mlir_op_trait!(Sin, OneOperand);
mlir_op_trait!(Sin, OneResult);
mlir_op_trait!(Sin, ZeroRegions);
mlir_op_trait!(Sin, ZeroSuccessors);

/// Constructs a new detached/owned [`SinOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`SinOperation`] for more information on the operation semantics.
pub fn sin<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedSinOperation<'c, 't>, Error> {
    sin_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`SinOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`SinOperation`] for more information on the operation semantics.
pub fn sin_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedSinOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.sin")?;
    let mut builder = OperationBuilder::new("math.sin", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::sin`"))
    })
}
/// Math [`Operation`] that computes a hyperbolic sine. Applies element-wise to scalar, vector, or tensor operands. The
/// result has the operand type.
///
/// # Example
///
/// The following is an example of a [`SinhOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.sinh %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathsinh-mathsinhop
pub trait SinhOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.sinh`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Sinh);
mlir_op_trait!(Sinh, OneOperand);
mlir_op_trait!(Sinh, OneResult);
mlir_op_trait!(Sinh, ZeroRegions);
mlir_op_trait!(Sinh, ZeroSuccessors);

/// Constructs a new detached/owned [`SinhOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`SinhOperation`] for more information on the operation semantics.
pub fn sinh<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedSinhOperation<'c, 't>, Error> {
    sinh_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`SinhOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`SinhOperation`] for more information on the operation semantics.
pub fn sinh_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedSinhOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.sinh")?;
    let mut builder = OperationBuilder::new("math.sinh", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::sinh`"))
    })
}

/// Math [`Operation`] that computes sine and cosine together. Returns sine first and cosine second, both with the input
/// type. Applies element-wise to scalar, vector, or tensor operands.
///
/// # Example
///
/// The following is an example of a [`SincosOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %sine, %cosine = math.sincos %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathsincos-mathsincosop
pub trait SincosOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the sine result.
    fn sine(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the cosine result.
    fn cosine(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(1)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.sincos`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Sincos);
mlir_op_trait!(Sincos, ZeroRegions);
mlir_op_trait!(Sincos, ZeroSuccessors);

/// Constructs a new detached/owned [`SincosOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`SincosOperation`] for more information on the operation semantics.
pub fn sincos<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedSincosOperation<'c, 't>, Error> {
    sincos_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`SincosOperation`] with explicit options at the specified [`Location`]. Refer to
/// the documentation of [`SincosOperation`] for more information on the operation semantics.
pub fn sincos_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedSincosOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.sincos")?;
    let mut builder = OperationBuilder::new("math.sincos", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder
        .add_operand(input)?
        .add_result(result_type)?
        .add_result(result_type)?
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::sincos`"))
        })
}

/// Math [`Operation`] that counts leading zero bits. Applies element-wise to scalar, vector, or tensor operands. The
/// result has the operand type.
///
/// # Example
///
/// The following is an example of a [`CountLeadingZerosOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %result = math.ctlz %input : i32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathctlz-mathcountleadingzerosop
pub trait CountLeadingZerosOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(CountLeadingZeros);
mlir_op_trait!(CountLeadingZeros, OneOperand);
mlir_op_trait!(CountLeadingZeros, OneResult);
mlir_op_trait!(CountLeadingZeros, ZeroRegions);
mlir_op_trait!(CountLeadingZeros, ZeroSuccessors);

/// Constructs a new detached/owned [`CountLeadingZerosOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`CountLeadingZerosOperation`] for more information on the operation semantics.
pub fn count_leading_zeros<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedCountLeadingZerosOperation<'c, 't>, Error> {
    count_leading_zeros_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`CountLeadingZerosOperation`] with explicit options at the specified [`Location`].
/// Refer to the documentation of [`CountLeadingZerosOperation`] for more information on the operation semantics.
pub fn count_leading_zeros_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedCountLeadingZerosOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_integer_like(result_type, "math.ctlz")?;
    let builder = OperationBuilder::new("math.ctlz", location);
    if options.fastmath != FastMathFlags::NONE {
        return Err(Error::invalid_argument("fast-math flags are unsupported for integer Math operations"));
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `math::count_leading_zeros`"))
    })
}
/// Math [`Operation`] that counts trailing zero bits. Applies element-wise to scalar, vector, or tensor operands. The
/// result has the operand type.
///
/// # Example
///
/// The following is an example of a [`CountTrailingZerosOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.cttz %input : i32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathcttz-mathcounttrailingzerosop
pub trait CountTrailingZerosOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(CountTrailingZeros);
mlir_op_trait!(CountTrailingZeros, OneOperand);
mlir_op_trait!(CountTrailingZeros, OneResult);
mlir_op_trait!(CountTrailingZeros, ZeroRegions);
mlir_op_trait!(CountTrailingZeros, ZeroSuccessors);

/// Constructs a new detached/owned [`CountTrailingZerosOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`CountTrailingZerosOperation`] for more information on the operation semantics.
pub fn count_trailing_zeros<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedCountTrailingZerosOperation<'c, 't>, Error> {
    count_trailing_zeros_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`CountTrailingZerosOperation`] with explicit options at the specified [`Location`].
/// Refer to the documentation of [`CountTrailingZerosOperation`] for more information on the operation semantics.
pub fn count_trailing_zeros_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedCountTrailingZerosOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_integer_like(result_type, "math.cttz")?;
    let builder = OperationBuilder::new("math.cttz", location);
    if options.fastmath != FastMathFlags::NONE {
        return Err(Error::invalid_argument("fast-math flags are unsupported for integer Math operations"));
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `math::count_trailing_zeros`"))
    })
}
/// Math [`Operation`] that counts set bits. Applies element-wise to scalar, vector, or tensor operands. The result has
/// the operand type.
///
/// # Example
///
/// The following is an example of a [`CtPopOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.ctpop %input : i32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathctpop-mathctpopop
pub trait CtPopOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(CtPop);
mlir_op_trait!(CtPop, OneOperand);
mlir_op_trait!(CtPop, OneResult);
mlir_op_trait!(CtPop, ZeroRegions);
mlir_op_trait!(CtPop, ZeroSuccessors);

/// Constructs a new detached/owned [`CtPopOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`CtPopOperation`] for more information on the operation semantics.
pub fn count_set_bits<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedCtPopOperation<'c, 't>, Error> {
    count_set_bits_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`CtPopOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`CtPopOperation`] for more information on the operation semantics.
pub fn count_set_bits_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedCtPopOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_integer_like(result_type, "math.ctpop")?;
    let builder = OperationBuilder::new("math.ctpop", location);
    if options.fastmath != FastMathFlags::NONE {
        return Err(Error::invalid_argument("fast-math flags are unsupported for integer Math operations"));
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `math::count_set_bits`"))
    })
}
/// Math [`Operation`] that computes the error function. Applies element-wise to scalar, vector, or tensor operands. The
/// result has the operand type.
///
/// # Example
///
/// The following is an example of a [`ErfOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.erf %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#matherf-matherfop
pub trait ErfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.erf`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Erf);
mlir_op_trait!(Erf, OneOperand);
mlir_op_trait!(Erf, OneResult);
mlir_op_trait!(Erf, ZeroRegions);
mlir_op_trait!(Erf, ZeroSuccessors);

/// Constructs a new detached/owned [`ErfOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`ErfOperation`] for more information on the operation semantics.
pub fn erf<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedErfOperation<'c, 't>, Error> {
    erf_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`ErfOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`ErfOperation`] for more information on the operation semantics.
pub fn erf_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedErfOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.erf")?;
    let mut builder = OperationBuilder::new("math.erf", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::erf`"))
    })
}
/// Math [`Operation`] that computes the complementary error function. Applies element-wise to scalar, vector, or tensor
/// operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`ErfcOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.erfc %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#matherfc-matherfcop
pub trait ErfcOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.erfc`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Erfc);
mlir_op_trait!(Erfc, OneOperand);
mlir_op_trait!(Erfc, OneResult);
mlir_op_trait!(Erfc, ZeroRegions);
mlir_op_trait!(Erfc, ZeroSuccessors);

/// Constructs a new detached/owned [`ErfcOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`ErfcOperation`] for more information on the operation semantics.
pub fn erfc<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedErfcOperation<'c, 't>, Error> {
    erfc_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`ErfcOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`ErfcOperation`] for more information on the operation semantics.
pub fn erfc_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedErfcOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.erfc")?;
    let mut builder = OperationBuilder::new("math.erfc", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::erfc`"))
    })
}
/// Math [`Operation`] that computes a base-e exponential. Applies element-wise to scalar, vector, or tensor operands.
/// The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`ExpOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.exp %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathexp-mathexpop
pub trait ExpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.exp`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Exp);
mlir_op_trait!(Exp, OneOperand);
mlir_op_trait!(Exp, OneResult);
mlir_op_trait!(Exp, ZeroRegions);
mlir_op_trait!(Exp, ZeroSuccessors);

/// Constructs a new detached/owned [`ExpOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`ExpOperation`] for more information on the operation semantics.
pub fn exp<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedExpOperation<'c, 't>, Error> {
    exp_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`ExpOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`ExpOperation`] for more information on the operation semantics.
pub fn exp_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedExpOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.exp")?;
    let mut builder = OperationBuilder::new("math.exp", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::exp`"))
    })
}
/// Math [`Operation`] that computes a base-two exponential. Applies element-wise to scalar, vector, or tensor operands.
/// The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`Exp2Operation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.exp2 %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathexp2-mathexp2op
pub trait Exp2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.exp2`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Exp2);
mlir_op_trait!(Exp2, OneOperand);
mlir_op_trait!(Exp2, OneResult);
mlir_op_trait!(Exp2, ZeroRegions);
mlir_op_trait!(Exp2, ZeroSuccessors);

/// Constructs a new detached/owned [`Exp2Operation`] at the specified [`Location`]. Refer to the documentation of
/// [`Exp2Operation`] for more information on the operation semantics.
pub fn exp2<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedExp2Operation<'c, 't>, Error> {
    exp2_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`Exp2Operation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`Exp2Operation`] for more information on the operation semantics.
pub fn exp2_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedExp2Operation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.exp2")?;
    let mut builder = OperationBuilder::new("math.exp2", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::exp2`"))
    })
}
/// Math [`Operation`] that computes one less than a base-e exponential. Applies element-wise to scalar, vector, or
/// tensor operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`ExpM1Operation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.expm1 %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathexpm1-mathexpm1op
pub trait ExpM1Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.expm1`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(ExpM1);
mlir_op_trait!(ExpM1, OneOperand);
mlir_op_trait!(ExpM1, OneResult);
mlir_op_trait!(ExpM1, ZeroRegions);
mlir_op_trait!(ExpM1, ZeroSuccessors);

/// Constructs a new detached/owned [`ExpM1Operation`] at the specified [`Location`]. Refer to the documentation of
/// [`ExpM1Operation`] for more information on the operation semantics.
pub fn expm1<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedExpM1Operation<'c, 't>, Error> {
    expm1_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`ExpM1Operation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`ExpM1Operation`] for more information on the operation semantics.
pub fn expm1_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedExpM1Operation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.expm1")?;
    let mut builder = OperationBuilder::new("math.expm1", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::expm1`"))
    })
}
/// Math [`Operation`] that computes a floor. The result is the greatest integral value less than or equal to the input.
/// Applies element-wise to scalar, vector, or tensor operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`FloorOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.floor %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathfloor-mathfloorop
pub trait FloorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.floor`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Floor);
mlir_op_trait!(Floor, OneOperand);
mlir_op_trait!(Floor, OneResult);
mlir_op_trait!(Floor, ZeroRegions);
mlir_op_trait!(Floor, ZeroSuccessors);

/// Constructs a new detached/owned [`FloorOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`FloorOperation`] for more information on the operation semantics.
pub fn floor<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedFloorOperation<'c, 't>, Error> {
    floor_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`FloorOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`FloorOperation`] for more information on the operation semantics.
pub fn floor_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedFloorOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.floor")?;
    let mut builder = OperationBuilder::new("math.floor", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::floor`"))
    })
}

/// Name of the rounding-mode attribute on floating-point Math operations that support it.
pub const ROUNDING_MODE_ATTRIBUTE: &str = "roundingmode";

/// Math [`Operation`] that computes a fused multiply-add. Computes `first * second + third` with a single rounding
/// step. Applies element-wise to scalar, vector, or tensor operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`FmaOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.fma %first, %second, %third : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathfma-mathfmaop
pub trait FmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the first input value.
    fn first(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the second input value.
    fn second(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the third input value.
    fn third(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.fma`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }

    /// Returns the explicit rounding mode, if present.
    fn rounding_mode(&self) -> Result<Option<RoundingMode>, Error> {
        self.attribute(ROUNDING_MODE_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<RoundingModeAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `roundingmode` attribute in `math.fma`"))?
                    .value()
                    .map(Some)
            })
            .unwrap_or(Ok(None))
    }
}

mlir_op!(Fma);
mlir_op_trait!(Fma, OneResult);
mlir_op_trait!(Fma, ZeroRegions);
mlir_op_trait!(Fma, ZeroSuccessors);

/// Constructs a new detached/owned [`FmaOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`FmaOperation`] for more information on the operation semantics.
pub fn fma<
    'first,
    'second,
    'third,
    'c: 'first + 'second + 'third,
    't: 'c,
    First: Value<'first, 'c, 't>,
    Second: Value<'second, 'c, 't>,
    Third: Value<'third, 'c, 't>,
    L: Location<'c, 't>,
>(
    first: First,
    second: Second,
    third: Third,
    location: L,
) -> Result<DetachedFmaOperation<'c, 't>, Error> {
    fma_with_options(first, second, third, Default::default(), location)
}

/// Constructs a new detached/owned [`FmaOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`FmaOperation`] for more information on the operation semantics.
pub fn fma_with_options<
    'first,
    'second,
    'third,
    'c: 'first + 'second + 'third,
    't: 'c,
    First: Value<'first, 'c, 't>,
    Second: Value<'second, 'c, 't>,
    Third: Value<'third, 'c, 't>,
    L: Location<'c, 't>,
>(
    first: First,
    second: Second,
    third: Third,
    options: FmaOptions,
    location: L,
) -> Result<DetachedFmaOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = first.r#type()?;
    if second.r#type()? != result_type || third.r#type()? != result_type {
        return Err(Error::invalid_argument("mismatched operand types for `math.fma`"));
    }
    validate_float_like(result_type, "math.fma")?;
    let mut builder = OperationBuilder::new("math.fma", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    if let Some(rounding_mode) = options.rounding_mode {
        builder =
            builder.add_attribute(ROUNDING_MODE_ATTRIBUTE, context.arith_rounding_mode_attribute(rounding_mode)?)?;
    }
    builder
        .add_operand(first)?
        .add_operand(second)?
        .add_operand(third)?
        .add_result(result_type)?
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::fma`"))
        })
}
/// Math [`Operation`] that computes an integer power. Applies element-wise to scalar, vector, or tensor operands. The
/// result has the operand type.
///
/// # Example
///
/// The following is an example of a [`IPowIOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.ipowi %lhs, %rhs : i32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathipowi-mathipowiop
pub trait IPowIOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand-side input value.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right-hand-side input value.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(IPowI);
mlir_op_trait!(IPowI, OneResult);
mlir_op_trait!(IPowI, ZeroRegions);
mlir_op_trait!(IPowI, ZeroSuccessors);

/// Constructs a new detached/owned [`IPowIOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`IPowIOperation`] for more information on the operation semantics.
pub fn ipowi<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    location: L,
) -> Result<DetachedIPowIOperation<'c, 't>, Error> {
    ipowi_with_options(lhs, rhs, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`IPowIOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`IPowIOperation`] for more information on the operation semantics.
pub fn ipowi_with_options<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedIPowIOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = lhs.r#type()?;
    if rhs.r#type()? != result_type {
        return Err(Error::invalid_argument("mismatched operand types for `math.ipowi`"));
    }
    validate_integer_like(result_type, "math.ipowi")?;
    let builder = OperationBuilder::new("math.ipowi", location);
    if options.fastmath != FastMathFlags::NONE {
        return Err(Error::invalid_argument("fast-math flags are unsupported for integer Math operations"));
    }
    builder
        .add_operand(lhs)?
        .add_operand(rhs)?
        .add_result(result_type)?
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::ipowi`"))
        })
}

/// Math [`Operation`] that tests whether values are finite. Applies element-wise to scalar, vector, or tensor operands.
/// The result has the input shape with `i1` elements.
///
/// # Example
///
/// The following is an example of a [`IsFiniteOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %result = math.isfinite %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathisfinite-mathisfiniteop
pub trait IsFiniteOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the classified input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the boolean classification result.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.isfinite`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(IsFinite);
mlir_op_trait!(IsFinite, OneOperand);
mlir_op_trait!(IsFinite, OneResult);
mlir_op_trait!(IsFinite, ZeroRegions);
mlir_op_trait!(IsFinite, ZeroSuccessors);

/// Constructs a new detached/owned [`IsFiniteOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`IsFiniteOperation`] for more information on the operation semantics.
pub fn is_finite<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedIsFiniteOperation<'c, 't>, Error> {
    is_finite_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`IsFiniteOperation`] with explicit options at the specified [`Location`]. Refer to
/// the documentation of [`IsFiniteOperation`] for more information on the operation semantics.
pub fn is_finite_with_options<'v, 'c: 'v, 't: 'c, V, L>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedIsFiniteOperation<'c, 't>, Error>
where
    V: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
{
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let input_type = input.r#type()?;
    validate_float_like(input_type, "math.isfinite")?;
    let result_type = boolean_type_like(context, input_type, location)?;
    let mut builder = OperationBuilder::new("math.isfinite", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::is_finite`"))
    })
}
/// Math [`Operation`] that tests whether values are infinite. Applies element-wise to scalar, vector, or tensor
/// operands. The result has the input shape with `i1` elements.
///
/// # Example
///
/// The following is an example of a [`IsInfOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.isinf %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathisinf-mathisinfop
pub trait IsInfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the classified input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the boolean classification result.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.isinf`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(IsInf);
mlir_op_trait!(IsInf, OneOperand);
mlir_op_trait!(IsInf, OneResult);
mlir_op_trait!(IsInf, ZeroRegions);
mlir_op_trait!(IsInf, ZeroSuccessors);

/// Constructs a new detached/owned [`IsInfOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`IsInfOperation`] for more information on the operation semantics.
pub fn is_infinite<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedIsInfOperation<'c, 't>, Error> {
    is_infinite_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`IsInfOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`IsInfOperation`] for more information on the operation semantics.
pub fn is_infinite_with_options<'v, 'c: 'v, 't: 'c, V, L>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedIsInfOperation<'c, 't>, Error>
where
    V: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
{
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let input_type = input.r#type()?;
    validate_float_like(input_type, "math.isinf")?;
    let result_type = boolean_type_like(context, input_type, location)?;
    let mut builder = OperationBuilder::new("math.isinf", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::is_infinite`"))
    })
}
/// Math [`Operation`] that tests whether values are NaN. Applies element-wise to scalar, vector, or tensor operands.
/// The result has the input shape with `i1` elements.
///
/// # Example
///
/// The following is an example of a [`IsNaNOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.isnan %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathisnan-mathisnanop
pub trait IsNaNOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the classified input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the boolean classification result.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.isnan`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(IsNaN);
mlir_op_trait!(IsNaN, OneOperand);
mlir_op_trait!(IsNaN, OneResult);
mlir_op_trait!(IsNaN, ZeroRegions);
mlir_op_trait!(IsNaN, ZeroSuccessors);

/// Constructs a new detached/owned [`IsNaNOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`IsNaNOperation`] for more information on the operation semantics.
pub fn is_nan<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedIsNaNOperation<'c, 't>, Error> {
    is_nan_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`IsNaNOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`IsNaNOperation`] for more information on the operation semantics.
pub fn is_nan_with_options<'v, 'c: 'v, 't: 'c, V, L>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedIsNaNOperation<'c, 't>, Error>
where
    V: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
{
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let input_type = input.r#type()?;
    validate_float_like(input_type, "math.isnan")?;
    let result_type = boolean_type_like(context, input_type, location)?;
    let mut builder = OperationBuilder::new("math.isnan", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::is_nan`"))
    })
}
/// Math [`Operation`] that tests whether values are normal. Normal values exclude zero, subnormal values, infinities,
/// and NaNs. Applies element-wise to scalar, vector, or tensor operands. The result has the input shape with `i1`
/// elements.
///
/// # Example
///
/// The following is an example of a [`IsNormalOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %result = math.isnormal %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathisnormal-mathisnormalop
pub trait IsNormalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the classified input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the boolean classification result.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.isnormal`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(IsNormal);
mlir_op_trait!(IsNormal, OneOperand);
mlir_op_trait!(IsNormal, OneResult);
mlir_op_trait!(IsNormal, ZeroRegions);
mlir_op_trait!(IsNormal, ZeroSuccessors);

/// Constructs a new detached/owned [`IsNormalOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`IsNormalOperation`] for more information on the operation semantics.
pub fn is_normal<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedIsNormalOperation<'c, 't>, Error> {
    is_normal_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`IsNormalOperation`] with explicit options at the specified [`Location`]. Refer to
/// the documentation of [`IsNormalOperation`] for more information on the operation semantics.
pub fn is_normal_with_options<'v, 'c: 'v, 't: 'c, V, L>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedIsNormalOperation<'c, 't>, Error>
where
    V: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
{
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let input_type = input.r#type()?;
    validate_float_like(input_type, "math.isnormal")?;
    let result_type = boolean_type_like(context, input_type, location)?;
    let mut builder = OperationBuilder::new("math.isnormal", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::is_normal`"))
    })
}
/// Math [`Operation`] that computes a natural logarithm. Applies element-wise to scalar, vector, or tensor operands.
/// The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`LogOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.log %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathlog-mathlogop
pub trait LogOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.log`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Log);
mlir_op_trait!(Log, OneOperand);
mlir_op_trait!(Log, OneResult);
mlir_op_trait!(Log, ZeroRegions);
mlir_op_trait!(Log, ZeroSuccessors);

/// Constructs a new detached/owned [`LogOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`LogOperation`] for more information on the operation semantics.
pub fn log<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedLogOperation<'c, 't>, Error> {
    log_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`LogOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`LogOperation`] for more information on the operation semantics.
pub fn log_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedLogOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.log")?;
    let mut builder = OperationBuilder::new("math.log", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::log`"))
    })
}
/// Math [`Operation`] that computes a base-ten logarithm. Applies element-wise to scalar, vector, or tensor operands.
/// The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`Log10Operation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.log10 %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathlog10-mathlog10op
pub trait Log10Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.log10`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Log10);
mlir_op_trait!(Log10, OneOperand);
mlir_op_trait!(Log10, OneResult);
mlir_op_trait!(Log10, ZeroRegions);
mlir_op_trait!(Log10, ZeroSuccessors);

/// Constructs a new detached/owned [`Log10Operation`] at the specified [`Location`]. Refer to the documentation of
/// [`Log10Operation`] for more information on the operation semantics.
pub fn log10<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedLog10Operation<'c, 't>, Error> {
    log10_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`Log10Operation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`Log10Operation`] for more information on the operation semantics.
pub fn log10_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedLog10Operation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.log10")?;
    let mut builder = OperationBuilder::new("math.log10", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::log10`"))
    })
}
/// Math [`Operation`] that computes the natural logarithm of one plus its input. Applies element-wise to scalar,
/// vector, or tensor operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`Log1pOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.log1p %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathlog1p-mathlog1pop
pub trait Log1pOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.log1p`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Log1p);
mlir_op_trait!(Log1p, OneOperand);
mlir_op_trait!(Log1p, OneResult);
mlir_op_trait!(Log1p, ZeroRegions);
mlir_op_trait!(Log1p, ZeroSuccessors);

/// Constructs a new detached/owned [`Log1pOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`Log1pOperation`] for more information on the operation semantics.
pub fn log1p<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedLog1pOperation<'c, 't>, Error> {
    log1p_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`Log1pOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`Log1pOperation`] for more information on the operation semantics.
pub fn log1p_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedLog1pOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.log1p")?;
    let mut builder = OperationBuilder::new("math.log1p", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::log1p`"))
    })
}
/// Math [`Operation`] that computes a base-two logarithm. Applies element-wise to scalar, vector, or tensor operands.
/// The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`Log2Operation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.log2 %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathlog2-mathlog2op
pub trait Log2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.log2`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Log2);
mlir_op_trait!(Log2, OneOperand);
mlir_op_trait!(Log2, OneResult);
mlir_op_trait!(Log2, ZeroRegions);
mlir_op_trait!(Log2, ZeroSuccessors);

/// Constructs a new detached/owned [`Log2Operation`] at the specified [`Location`]. Refer to the documentation of
/// [`Log2Operation`] for more information on the operation semantics.
pub fn log2<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedLog2Operation<'c, 't>, Error> {
    log2_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`Log2Operation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`Log2Operation`] for more information on the operation semantics.
pub fn log2_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedLog2Operation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.log2")?;
    let mut builder = OperationBuilder::new("math.log2", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::log2`"))
    })
}
/// Math [`Operation`] that computes a floating-point power. Applies element-wise to scalar, vector, or tensor operands.
/// The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`PowFOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.powf %lhs, %rhs : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathpowf-mathpowfop
pub trait PowFOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand-side input value.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right-hand-side input value.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.powf`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(PowF);
mlir_op_trait!(PowF, OneResult);
mlir_op_trait!(PowF, ZeroRegions);
mlir_op_trait!(PowF, ZeroSuccessors);

/// Constructs a new detached/owned [`PowFOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`PowFOperation`] for more information on the operation semantics.
pub fn powf<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    location: L,
) -> Result<DetachedPowFOperation<'c, 't>, Error> {
    powf_with_options(lhs, rhs, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`PowFOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`PowFOperation`] for more information on the operation semantics.
pub fn powf_with_options<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    Lhs: Value<'lhs, 'c, 't>,
    Rhs: Value<'rhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: Lhs,
    rhs: Rhs,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedPowFOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = lhs.r#type()?;
    if rhs.r#type()? != result_type {
        return Err(Error::invalid_argument("mismatched operand types for `math.powf`"));
    }
    validate_float_like(result_type, "math.powf")?;
    let mut builder = OperationBuilder::new("math.powf", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder
        .add_operand(lhs)?
        .add_operand(rhs)?
        .add_result(result_type)?
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::powf`"))
        })
}
/// Math [`Operation`] that computes a reciprocal square root. Applies element-wise to scalar, vector, or tensor
/// operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`RsqrtOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.rsqrt %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathrsqrt-mathrsqrtop
pub trait RsqrtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.rsqrt`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Rsqrt);
mlir_op_trait!(Rsqrt, OneOperand);
mlir_op_trait!(Rsqrt, OneResult);
mlir_op_trait!(Rsqrt, ZeroRegions);
mlir_op_trait!(Rsqrt, ZeroSuccessors);

/// Constructs a new detached/owned [`RsqrtOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`RsqrtOperation`] for more information on the operation semantics.
pub fn rsqrt<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedRsqrtOperation<'c, 't>, Error> {
    rsqrt_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`RsqrtOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`RsqrtOperation`] for more information on the operation semantics.
pub fn rsqrt_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedRsqrtOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.rsqrt")?;
    let mut builder = OperationBuilder::new("math.rsqrt", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::rsqrt`"))
    })
}
/// Math [`Operation`] that computes a square root. Applies element-wise to scalar, vector, or tensor operands. The
/// result has the operand type.
///
/// # Example
///
/// The following is an example of a [`SqrtOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.sqrt %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathsqrt-mathsqrtop
pub trait SqrtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.sqrt`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Sqrt);
mlir_op_trait!(Sqrt, OneOperand);
mlir_op_trait!(Sqrt, OneResult);
mlir_op_trait!(Sqrt, ZeroRegions);
mlir_op_trait!(Sqrt, ZeroSuccessors);

/// Constructs a new detached/owned [`SqrtOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`SqrtOperation`] for more information on the operation semantics.
pub fn sqrt<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedSqrtOperation<'c, 't>, Error> {
    sqrt_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`SqrtOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`SqrtOperation`] for more information on the operation semantics.
pub fn sqrt_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedSqrtOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.sqrt")?;
    let mut builder = OperationBuilder::new("math.sqrt", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::sqrt`"))
    })
}
/// Math [`Operation`] that computes a tangent. Applies element-wise to scalar, vector, or tensor operands. The result
/// has the operand type.
///
/// # Example
///
/// The following is an example of a [`TanOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.tan %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathtan-mathtanop
pub trait TanOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.tan`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Tan);
mlir_op_trait!(Tan, OneOperand);
mlir_op_trait!(Tan, OneResult);
mlir_op_trait!(Tan, ZeroRegions);
mlir_op_trait!(Tan, ZeroSuccessors);

/// Constructs a new detached/owned [`TanOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`TanOperation`] for more information on the operation semantics.
pub fn tan<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedTanOperation<'c, 't>, Error> {
    tan_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`TanOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`TanOperation`] for more information on the operation semantics.
pub fn tan_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedTanOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.tan")?;
    let mut builder = OperationBuilder::new("math.tan", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::tan`"))
    })
}
/// Math [`Operation`] that computes a hyperbolic tangent. Applies element-wise to scalar, vector, or tensor operands.
/// The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`TanhOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.tanh %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathtanh-mathtanhop
pub trait TanhOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.tanh`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Tanh);
mlir_op_trait!(Tanh, OneOperand);
mlir_op_trait!(Tanh, OneResult);
mlir_op_trait!(Tanh, ZeroRegions);
mlir_op_trait!(Tanh, ZeroSuccessors);

/// Constructs a new detached/owned [`TanhOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`TanhOperation`] for more information on the operation semantics.
pub fn tanh<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedTanhOperation<'c, 't>, Error> {
    tanh_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`TanhOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`TanhOperation`] for more information on the operation semantics.
pub fn tanh_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedTanhOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.tanh")?;
    let mut builder = OperationBuilder::new("math.tanh", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::tanh`"))
    })
}
/// Math [`Operation`] that rounds halfway cases to even. Ties round to the nearest even integer. Applies element-wise
/// to scalar, vector, or tensor operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`RoundEvenOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %result = math.roundeven %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathroundeven-mathroundevenop
pub trait RoundEvenOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.roundeven`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(RoundEven);
mlir_op_trait!(RoundEven, OneOperand);
mlir_op_trait!(RoundEven, OneResult);
mlir_op_trait!(RoundEven, ZeroRegions);
mlir_op_trait!(RoundEven, ZeroSuccessors);

/// Constructs a new detached/owned [`RoundEvenOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`RoundEvenOperation`] for more information on the operation semantics.
pub fn round_even<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedRoundEvenOperation<'c, 't>, Error> {
    round_even_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`RoundEvenOperation`] with explicit options at the specified [`Location`]. Refer to
/// the documentation of [`RoundEvenOperation`] for more information on the operation semantics.
pub fn round_even_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedRoundEvenOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.roundeven")?;
    let mut builder = OperationBuilder::new("math.roundeven", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::round_even`"))
    })
}
/// Math [`Operation`] that rounds halfway cases away from zero. Ties round away from zero. Applies element-wise to
/// scalar, vector, or tensor operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`RoundOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.round %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathround-mathroundop
pub trait RoundOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.round`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Round);
mlir_op_trait!(Round, OneOperand);
mlir_op_trait!(Round, OneResult);
mlir_op_trait!(Round, ZeroRegions);
mlir_op_trait!(Round, ZeroSuccessors);

/// Constructs a new detached/owned [`RoundOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`RoundOperation`] for more information on the operation semantics.
pub fn round<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedRoundOperation<'c, 't>, Error> {
    round_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`RoundOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`RoundOperation`] for more information on the operation semantics.
pub fn round_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedRoundOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.round")?;
    let mut builder = OperationBuilder::new("math.round", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::round`"))
    })
}
/// Math [`Operation`] that truncates a floating-point value. The result rounds toward zero. Applies element-wise to
/// scalar, vector, or tensor operands. The result has the operand type.
///
/// # Example
///
/// The following is an example of a [`TruncOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.trunc %input : f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathtrunc-mathtruncop
pub trait TruncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.trunc`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(Trunc);
mlir_op_trait!(Trunc, OneOperand);
mlir_op_trait!(Trunc, OneResult);
mlir_op_trait!(Trunc, ZeroRegions);
mlir_op_trait!(Trunc, ZeroSuccessors);

/// Constructs a new detached/owned [`TruncOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`TruncOperation`] for more information on the operation semantics.
pub fn trunc<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> Result<DetachedTruncOperation<'c, 't>, Error> {
    trunc_with_options(input, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`TruncOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`TruncOperation`] for more information on the operation semantics.
pub fn trunc_with_options<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedTruncOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = input.r#type()?;
    validate_float_like(result_type, "math.trunc")?;
    let mut builder = OperationBuilder::new("math.trunc", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder.add_operand(input)?.add_result(result_type)?.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::trunc`"))
    })
}

/// Math [`Operation`] that raises a floating-point base to an integer power. The base and integer exponent have
/// matching shapes; the result has the base type. Applies element-wise to scalar, vector, or tensor operands.
///
/// # Example
///
/// The following is an example of a [`FPowIOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = math.fpowi %base, %power : f32, i32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/MathOps/#mathfpowi-mathfpowiop
pub trait FPowIOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the floating-point base.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the integer power.
    fn power(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the fast-math flags.
    fn fastmath(&self) -> Result<FastMathFlags, Error> {
        self.attribute(FASTMATH_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<FastMathFlagsAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `fastmath` attribute in `math.fpowi`"))?
                    .value()
            })
            .unwrap_or(Ok(FastMathFlags::NONE))
    }
}

mlir_op!(FPowI);
mlir_op_trait!(FPowI, OneResult);
mlir_op_trait!(FPowI, ZeroRegions);
mlir_op_trait!(FPowI, ZeroSuccessors);

/// Constructs a new detached/owned [`FPowIOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`FPowIOperation`] for more information on the operation semantics.
pub fn fpowi<
    'base,
    'power,
    'c: 'base + 'power,
    't: 'c,
    Base: Value<'base, 'c, 't>,
    Power: Value<'power, 'c, 't>,
    L: Location<'c, 't>,
>(
    base: Base,
    power: Power,
    location: L,
) -> Result<DetachedFPowIOperation<'c, 't>, Error> {
    fpowi_with_options(base, power, MathOperationOptions::default(), location)
}

/// Constructs a new detached/owned [`FPowIOperation`] with explicit options at the specified [`Location`]. Refer to the
/// documentation of [`FPowIOperation`] for more information on the operation semantics.
pub fn fpowi_with_options<
    'base,
    'power,
    'c: 'base + 'power,
    't: 'c,
    Base: Value<'base, 'c, 't>,
    Power: Value<'power, 'c, 't>,
    L: Location<'c, 't>,
>(
    base: Base,
    power: Power,
    options: MathOperationOptions,
    location: L,
) -> Result<DetachedFPowIOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::math()?)?;
    let result_type = base.r#type()?;
    let power_type = power.r#type()?;
    validate_float_like(result_type, "math.fpowi")?;
    validate_integer_like(power_type, "math.fpowi")?;
    validate_same_shape(result_type, power_type, "math.fpowi")?;
    let mut builder = OperationBuilder::new("math.fpowi", location);
    if options.fastmath != FastMathFlags::NONE {
        builder =
            builder.add_attribute(FASTMATH_ATTRIBUTE, context.arith_fast_math_flags_attribute(options.fastmath)?)?;
    }
    builder
        .add_operand(base)?
        .add_operand(power)?
        .add_result(result_type)?
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `math::fpowi`"))
        })
}

/// Returns the scalar element type of a supported shaped value, or the type itself for a scalar.
fn element_type<'c, 't: 'c>(r#type: TypeRef<'c, 't>) -> Result<TypeRef<'c, 't>, Error> {
    if let Some(r#type) = r#type.cast::<VectorTypeRef>() {
        r#type.element_type()
    } else if let Some(r#type) = r#type.cast::<TensorTypeRef>() {
        r#type.element_type()
    } else if let Some(r#type) = r#type.cast::<UnrankedTensorTypeRef>() {
        r#type.element_type()
    } else {
        Ok(r#type)
    }
}

/// Checks that the scalar or shaped operand has floating-point elements.
fn validate_float_like(r#type: TypeRef<'_, '_>, operation_name: &str) -> Result<(), Error> {
    if element_type(r#type)?.is::<FloatTypeRef>() {
        Ok(())
    } else {
        Err(Error::invalid_argument(format!("expected floating-point operands for `{operation_name}`")))
    }
}

/// Checks that the scalar or shaped operand has integer or index elements.
fn validate_integer_like(r#type: TypeRef<'_, '_>, operation_name: &str) -> Result<(), Error> {
    let element_type = element_type(r#type)?;
    if element_type.is::<IntegerTypeRef>() || element_type.is::<IndexTypeRef>() {
        Ok(())
    } else {
        Err(Error::invalid_argument(format!("expected integer or index operands for `{operation_name}`")))
    }
}

/// Checks scalar/shaped categories and dimensions while allowing different element types.
fn validate_same_shape(lhs: TypeRef<'_, '_>, rhs: TypeRef<'_, '_>, operation_name: &str) -> Result<(), Error> {
    let matching_shape = if let (Some(lhs), Some(rhs)) = (lhs.cast::<VectorTypeRef>(), rhs.cast::<VectorTypeRef>()) {
        lhs.dimensions().eq(rhs.dimensions())
    } else if let (Some(lhs), Some(rhs)) = (lhs.cast::<TensorTypeRef>(), rhs.cast::<TensorTypeRef>()) {
        lhs.dimensions().eq(rhs.dimensions())
    } else if lhs.is::<UnrankedTensorTypeRef>() && rhs.is::<UnrankedTensorTypeRef>() {
        true
    } else {
        !lhs.is::<VectorTypeRef>()
            && !rhs.is::<VectorTypeRef>()
            && !lhs.is::<TensorTypeRef>()
            && !rhs.is::<TensorTypeRef>()
            && !lhs.is::<UnrankedTensorTypeRef>()
            && !rhs.is::<UnrankedTensorTypeRef>()
    };
    if matching_shape {
        Ok(())
    } else {
        Err(Error::invalid_argument(format!("mismatched operand shapes for `{operation_name}`")))
    }
}

/// Constructs a boolean result type preserving the input shape and tensor encoding.
fn boolean_type_like<'c, 't: 'c, L: Location<'c, 't>>(
    context: &'c crate::Context<'t>,
    input_type: TypeRef<'c, 't>,
    location: L,
) -> Result<TypeRef<'c, 't>, Error> {
    let boolean_type = context.signless_integer_type(1);
    if let Some(input_type) = input_type.cast::<VectorTypeRef>() {
        let dimensions = input_type.dimensions().collect::<Vec<VectorTypeDimension>>();
        Ok(context.vector_type(boolean_type, &dimensions, location)?.as_ref())
    } else if let Some(input_type) = input_type.cast::<TensorTypeRef>() {
        let dimensions = input_type.dimensions().collect::<Vec<_>>();
        Ok(context.tensor_type(boolean_type, &dimensions, input_type.encoding()?, location)?.as_ref())
    } else if input_type.is::<UnrankedTensorTypeRef>() {
        Ok(context.unranked_tensor_type(boolean_type, location)?.as_ref())
    } else {
        Ok(boolean_type.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, Operation, Type, Value};

    use super::*;

    macro_rules! math_assert_unary_contract {
        // Generates the shared assertions for this operation family.
        (false, $function_name:ident, $operation_name:literal, $input:expr, $location:expr, $context:expr) => {
            paste::paste! {
                let operation = [<$function_name _with_options>](
                    $input,
                    MathOperationOptions { fastmath: FastMathFlags::NO_NANS },
                    $location,
                )
                .unwrap();
                assert_eq!(operation.fastmath().unwrap(), FastMathFlags::NO_NANS);
                let invalid_block = $context.block(&[($context.signless_integer_type(32).as_ref(), $location)]);
                assert!(matches!(
                    $function_name(invalid_block.argument(0).unwrap(), $location),
                    Err(Error::InvalidArgument { message, .. })
                        if message == concat!("expected floating-point operands for `", $operation_name, "`"),
                ));
            }
        };
        // Handles the alternate operand or options category.
        (true, $function_name:ident, $operation_name:literal, $input:expr, $location:expr, $context:expr) => {
            let invalid_block = $context.block(&[($context.float32_type().as_ref(), $location)]);
            assert!(matches!(
                $function_name(invalid_block.argument(0).unwrap(), $location),
                Err(Error::InvalidArgument { message, .. })
                    if message == concat!("expected integer or index operands for `", $operation_name, "`"),
            ));
        };
    }

    macro_rules! math_assert_binary_contract {
        // Generates the shared assertions for this operation family.
        (false, $function_name:ident, $lhs:expr, $rhs:expr, $location:expr, $context:expr) => {
            paste::paste! {
                let operation = [<$function_name _with_options>](
                    $lhs,
                    $rhs,
                    MathOperationOptions { fastmath: FastMathFlags::NO_NANS },
                    $location,
                )
                .unwrap();
                assert_eq!(operation.fastmath().unwrap(), FastMathFlags::NO_NANS);
            }
        };
        // Handles the alternate operand or options category.
        (true, $function_name:ident, $lhs:expr, $rhs:expr, $location:expr, $context:expr) => {
            let invalid_block = $context.block(&[
                ($context.signless_integer_type(32).as_ref(), $location),
                ($context.float32_type().as_ref(), $location),
            ]);
            assert!(matches!(
                $function_name(invalid_block.argument(0).unwrap(), invalid_block.argument(1).unwrap(), $location),
                Err(Error::InvalidArgument { message, .. })
                    if message == "mismatched operand types for `math.ipowi`",
            ));
        };
    }

    macro_rules! math_assert_ternary_contract {
        // Generates the shared assertions for this operation family.
        (fma, $first:expr, $second:expr, $third:expr, $location:expr) => {
            let operation = fma_with_options(
                $first,
                $second,
                $third,
                FmaOptions { fastmath: FastMathFlags::NO_NANS, rounding_mode: Some(RoundingMode::TowardZero) },
                $location,
            )
            .unwrap();
            assert_eq!(operation.fastmath().unwrap(), FastMathFlags::NO_NANS);
            assert_eq!(operation.rounding_mode().unwrap(), Some(RoundingMode::TowardZero));
        };
        // Handles the alternate operand or options category.
        ($function_name:ident, $first:expr, $second:expr, $third:expr, $location:expr) => {
            paste::paste! {
                let operation = [<$function_name _with_options>](
                    $first,
                    $second,
                    $third,
                    MathOperationOptions { fastmath: FastMathFlags::NO_NANS },
                    $location,
                )
                .unwrap();
                assert_eq!(operation.fastmath().unwrap(), FastMathFlags::NO_NANS);
            }
        };
    }

    macro_rules! math_test_unary_operation {
        // Generates the shared assertions for this operation family.
        ($test_name:ident, $function_name:ident, $operation_name:literal, $integer:tt) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let value_type =
                    if $integer { context.signless_integer_type(32).as_ref() } else { context.float32_type().as_ref() };
                let module = context.module(location).unwrap();
                module
                    .body()
                    .unwrap()
                    .append_operation({
                        let mut block = context.block(&[(value_type, location)]);
                        let operation = $function_name(block.argument(0).unwrap(), location).unwrap();
                        assert_eq!(operation.input().unwrap(), block.argument(0).unwrap());
                        assert_eq!(operation.result_value().unwrap().r#type().unwrap(), value_type);
                        math_assert_unary_contract!(
                            $integer,
                            $function_name,
                            $operation_name,
                            block.argument(0).unwrap(),
                            location,
                            context
                        );
                        let operation = block.append_operation(operation).unwrap();
                        block
                            .append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap())
                            .unwrap();
                        func::func(
                            stringify!($test_name).strip_prefix("test_").unwrap(),
                            func::FuncAttributes {
                                arguments: vec![value_type.into()],
                                results: vec![value_type.into()],
                                ..Default::default()
                            },
                            block.try_into().unwrap(),
                            location,
                        )
                        .unwrap()
                    })
                    .unwrap();
                assert_eq!(module.verify(), Ok(true));
                let type_name = if $integer { "i32" } else { "f32" };
                let function_name = stringify!($test_name).strip_prefix("test_").unwrap();
                assert_eq!(
                    module.to_string(),
                    format!(
                        indoc! {"
                        module {{
                          func.func @{function_name}(%arg0: {type_name}) -> {type_name} {{
                            %0 = {} %arg0 : {type_name}
                            return %0 : {type_name}
                          }}
                        }}
                    "},
                        $operation_name,
                        function_name = function_name,
                        type_name = type_name,
                    ),
                );
            }
        };
    }

    macro_rules! math_test_binary_operation {
        // Generates the shared assertions for this operation family.
        ($test_name:ident, $function_name:ident, $operation_name:literal, $integer:tt) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let value_type =
                    if $integer { context.signless_integer_type(32).as_ref() } else { context.float32_type().as_ref() };
                let module = context.module(location).unwrap();
                module
                    .body()
                    .unwrap()
                    .append_operation({
                        let mut block = context.block(&[(value_type, location), (value_type, location)]);
                        let operation =
                            $function_name(block.argument(0).unwrap(), block.argument(1).unwrap(), location).unwrap();
                        assert_eq!(operation.lhs().unwrap(), block.argument(0).unwrap());
                        assert_eq!(operation.rhs().unwrap(), block.argument(1).unwrap());
                        assert_eq!(operation.result_value().unwrap().r#type().unwrap(), value_type);
                        math_assert_binary_contract!(
                            $integer,
                            $function_name,
                            block.argument(0).unwrap(),
                            block.argument(1).unwrap(),
                            location,
                            context
                        );
                        let operation = block.append_operation(operation).unwrap();
                        block
                            .append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap())
                            .unwrap();
                        func::func(
                            stringify!($test_name).strip_prefix("test_").unwrap(),
                            func::FuncAttributes {
                                arguments: vec![value_type.into(), value_type.into()],
                                results: vec![value_type.into()],
                                ..Default::default()
                            },
                            block.try_into().unwrap(),
                            location,
                        )
                        .unwrap()
                    })
                    .unwrap();
                assert_eq!(module.verify(), Ok(true));
                let type_name = if $integer { "i32" } else { "f32" };
                let function_name = stringify!($test_name).strip_prefix("test_").unwrap();
                assert_eq!(
                    module.to_string(),
                    format!(
                        indoc! {"
                        module {{
                          func.func @{function_name}(%arg0: {type_name}, %arg1: {type_name}) -> {type_name} {{
                            %0 = {} %arg0, %arg1 : {type_name}
                            return %0 : {type_name}
                          }}
                        }}
                    "},
                        $operation_name,
                        function_name = function_name,
                        type_name = type_name,
                    ),
                );
            }
        };
    }

    macro_rules! math_test_ternary_operation {
        // Generates the shared assertions for this operation family.
        ($test_name:ident, $function_name:ident, $rendering:literal) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let value_type = context.float32_type();
                let module = context.module(location).unwrap();
                module
                    .body()
                    .unwrap()
                    .append_operation({
                        let mut block =
                            context.block(&[(value_type, location), (value_type, location), (value_type, location)]);
                        let operation = $function_name(
                            block.argument(0).unwrap(),
                            block.argument(1).unwrap(),
                            block.argument(2).unwrap(),
                            location,
                        )
                        .unwrap();
                        assert_eq!(operation.first().unwrap(), block.argument(0).unwrap());
                        assert_eq!(operation.second().unwrap(), block.argument(1).unwrap());
                        assert_eq!(operation.third().unwrap(), block.argument(2).unwrap());
                        math_assert_ternary_contract!(
                            $function_name,
                            block.argument(0).unwrap(),
                            block.argument(1).unwrap(),
                            block.argument(2).unwrap(),
                            location
                        );
                        let operation = block.append_operation(operation).unwrap();
                        block
                            .append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap())
                            .unwrap();
                        func::func(
                            stringify!($test_name).strip_prefix("test_").unwrap(),
                            func::FuncAttributes {
                                arguments: vec![value_type.into(), value_type.into(), value_type.into()],
                                results: vec![value_type.into()],
                                ..Default::default()
                            },
                            block.try_into().unwrap(),
                            location,
                        )
                        .unwrap()
                    })
                    .unwrap();
                assert_eq!(module.verify(), Ok(true));
                let function_name = stringify!($test_name).strip_prefix("test_").unwrap();
                assert_eq!(
                    module.to_string(),
                    format!(
                        indoc! {"
                        module {{
                          func.func @{function_name}(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {{
                            %0 = {}
                            return %0 : f32
                          }}
                        }}
                    "},
                        $rendering,
                        function_name = function_name,
                    ),
                );
            }
        };
    }

    math_test_unary_operation!(test_absf, absf, "math.absf", false);
    math_test_unary_operation!(test_absi, absi, "math.absi", true);
    math_test_unary_operation!(test_acosh, acosh, "math.acosh", false);
    math_test_unary_operation!(test_asin, asin, "math.asin", false);
    math_test_unary_operation!(test_asinh, asinh, "math.asinh", false);
    math_test_unary_operation!(test_atan, atan, "math.atan", false);
    math_test_unary_operation!(test_atanh, atanh, "math.atanh", false);
    math_test_binary_operation!(test_atan2, atan2, "math.atan2", false);
    math_test_unary_operation!(test_cbrt, cbrt, "math.cbrt", false);
    math_test_unary_operation!(test_ceil, ceil, "math.ceil", false);
    math_test_ternary_operation!(test_clampf, clampf, "math.clampf %arg0 to [%arg1, %arg2] : f32");
    math_test_binary_operation!(test_copysign, copysign, "math.copysign", false);
    math_test_unary_operation!(test_cos, cos, "math.cos", false);
    math_test_unary_operation!(test_acos, acos, "math.acos", false);
    math_test_unary_operation!(test_cosh, cosh, "math.cosh", false);
    math_test_unary_operation!(test_sin, sin, "math.sin", false);
    math_test_unary_operation!(test_sinh, sinh, "math.sinh", false);

    #[test]
    fn test_sincos() {
        let context = Context::new();
        let location = context.unknown_location();
        let value_type = context.float32_type();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(value_type, location)]);
                let operation = sincos(block.argument(0).unwrap(), location).unwrap();
                assert_eq!(operation.input().unwrap(), block.argument(0).unwrap());
                assert_eq!(operation.sine().unwrap().r#type().unwrap(), value_type);
                assert_eq!(operation.cosine().unwrap().r#type().unwrap(), value_type);
                let configured = sincos_with_options(
                    block.argument(0).unwrap(),
                    MathOperationOptions { fastmath: FastMathFlags::NO_NANS },
                    location,
                )
                .unwrap();
                assert_eq!(configured.fastmath().unwrap(), FastMathFlags::NO_NANS);
                let invalid_block = context.block(&[(context.signless_integer_type(32).as_ref(), location)]);
                assert!(matches!(
                    sincos(invalid_block.argument(0).unwrap(), location),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "expected floating-point operands for `math.sincos`",
                ));
                let operation = block.append_operation(operation).unwrap();
                block
                    .append_operation(
                        func::r#return(&[operation.result(0).unwrap(), operation.result(1).unwrap()], location)
                            .unwrap(),
                    )
                    .unwrap();
                func::func(
                    "sincos",
                    func::FuncAttributes {
                        arguments: vec![value_type.into()],
                        results: vec![value_type.into(), value_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
                        module {
                          func.func @sincos(%arg0: f32) -> (f32, f32) {
                            %sin, %cos = math.sincos %arg0 : f32
                            return %sin, %cos : f32, f32
                          }
                        }
                    "},
        );
    }

    math_test_unary_operation!(test_count_leading_zeros, count_leading_zeros, "math.ctlz", true);
    math_test_unary_operation!(test_count_trailing_zeros, count_trailing_zeros, "math.cttz", true);
    math_test_unary_operation!(test_count_set_bits, count_set_bits, "math.ctpop", true);
    math_test_unary_operation!(test_erf, erf, "math.erf", false);
    math_test_unary_operation!(test_erfc, erfc, "math.erfc", false);
    math_test_unary_operation!(test_exp, exp, "math.exp", false);
    math_test_unary_operation!(test_exp2, exp2, "math.exp2", false);
    math_test_unary_operation!(test_expm1, expm1, "math.expm1", false);
    math_test_unary_operation!(test_floor, floor, "math.floor", false);
    math_test_ternary_operation!(test_fma, fma, "math.fma %arg0, %arg1, %arg2 : f32");
    math_test_binary_operation!(test_ipowi, ipowi, "math.ipowi", true);

    macro_rules! math_test_classification_operation {
        // Generates the shared assertions for this operation family.
        ($test_name:ident, $function_name:ident, $operation_name:literal) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let input_type = context.float32_type();
                let result_type = context.signless_integer_type(1);
                let module = context.module(location).unwrap();
                module
                    .body()
                    .unwrap()
                    .append_operation({
                        let mut block = context.block(&[(input_type, location)]);
                        let operation = $function_name(block.argument(0).unwrap(), location).unwrap();
                        assert_eq!(operation.input().unwrap(), block.argument(0).unwrap());
                        assert_eq!(operation.result_value().unwrap().r#type().unwrap(), result_type);
                        paste::paste! {
                            let configured = [<$function_name _with_options>](
                                block.argument(0).unwrap(),
                                MathOperationOptions { fastmath: FastMathFlags::NO_NANS },
                                location,
                            )
                            .unwrap();
                            assert_eq!(configured.fastmath().unwrap(), FastMathFlags::NO_NANS);
                        }
                        let vector_type =
                            context.vector_type(input_type, &[VectorTypeDimension::Fixed(4)], location).unwrap();
                        let vector_block = context.block(&[(vector_type.as_ref(), location)]);
                        let vector_operation = $function_name(vector_block.argument(0).unwrap(), location).unwrap();
                        assert_eq!(
                            vector_operation.result_value().unwrap().r#type().unwrap().to_string(),
                            "vector<4xi1>",
                        );
                        let invalid_block = context.block(&[(context.signless_integer_type(32).as_ref(), location)]);
                        assert!(matches!(
                            $function_name(invalid_block.argument(0).unwrap(), location),
                            Err(Error::InvalidArgument { message, .. })
                                if message == concat!("expected floating-point operands for `", $operation_name, "`"),
                        ));
                        let operation = block.append_operation(operation).unwrap();
                        block
                            .append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap())
                            .unwrap();
                        func::func(
                            stringify!($test_name).strip_prefix("test_").unwrap(),
                            func::FuncAttributes {
                                arguments: vec![input_type.into()],
                                results: vec![result_type.into()],
                                ..Default::default()
                            },
                            block.try_into().unwrap(),
                            location,
                        )
                        .unwrap()
                    })
                    .unwrap();
                assert_eq!(module.verify(), Ok(true));
                let function_name = stringify!($test_name).strip_prefix("test_").unwrap();
                assert_eq!(
                    module.to_string(),
                    format!(
                        indoc! {"
                        module {{
                          func.func @{function_name}(%arg0: f32) -> i1 {{
                            %0 = {} %arg0 : f32
                            return %0 : i1
                          }}
                        }}
                    "},
                        $operation_name,
                        function_name = function_name,
                    ),
                );
            }
        };
    }

    math_test_classification_operation!(test_is_finite, is_finite, "math.isfinite");
    math_test_classification_operation!(test_is_infinite, is_infinite, "math.isinf");
    math_test_classification_operation!(test_is_nan, is_nan, "math.isnan");
    math_test_classification_operation!(test_is_normal, is_normal, "math.isnormal");
    math_test_unary_operation!(test_log, log, "math.log", false);
    math_test_unary_operation!(test_log10, log10, "math.log10", false);
    math_test_unary_operation!(test_log1p, log1p, "math.log1p", false);
    math_test_unary_operation!(test_log2, log2, "math.log2", false);
    math_test_binary_operation!(test_powf, powf, "math.powf", false);
    math_test_unary_operation!(test_rsqrt, rsqrt, "math.rsqrt", false);
    math_test_unary_operation!(test_sqrt, sqrt, "math.sqrt", false);
    math_test_unary_operation!(test_tan, tan, "math.tan", false);
    math_test_unary_operation!(test_tanh, tanh, "math.tanh", false);
    math_test_unary_operation!(test_round_even, round_even, "math.roundeven", false);
    math_test_unary_operation!(test_round, round, "math.round", false);
    math_test_unary_operation!(test_trunc, trunc, "math.trunc", false);

    #[test]
    fn test_fpowi() {
        let context = Context::new();
        let location = context.unknown_location();
        let float_type = context.float32_type();
        let integer_type = context.signless_integer_type(32);
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(float_type.as_ref(), location), (integer_type.as_ref(), location)]);
                let operation = fpowi(block.argument(0).unwrap(), block.argument(1).unwrap(), location).unwrap();
                assert_eq!(operation.base().unwrap(), block.argument(0).unwrap());
                assert_eq!(operation.power().unwrap(), block.argument(1).unwrap());
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), float_type);
                let configured = fpowi_with_options(
                    block.argument(0).unwrap(),
                    block.argument(1).unwrap(),
                    MathOperationOptions { fastmath: FastMathFlags::NO_NANS },
                    location,
                )
                .unwrap();
                assert_eq!(configured.fastmath().unwrap(), FastMathFlags::NO_NANS);
                assert!(matches!(
                    fpowi(block.argument(1).unwrap(), block.argument(1).unwrap(), location),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "expected floating-point operands for `math.fpowi`",
                ));
                assert!(matches!(
                    fpowi(block.argument(0).unwrap(), block.argument(0).unwrap(), location),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "expected integer or index operands for `math.fpowi`",
                ));
                let operation = block.append_operation(operation).unwrap();
                block.append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "fpowi",
                    func::FuncAttributes {
                        arguments: vec![float_type.into(), integer_type.into()],
                        results: vec![float_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
                        module {
                          func.func @fpowi(%arg0: f32, %arg1: i32) -> f32 {
                            %0 = math.fpowi %arg0, %arg1 : f32, i32
                            return %0 : f32
                          }
                        }
                    "},
        );
    }
}
