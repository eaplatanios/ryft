use std::fmt::Display;
use std::ops::{Add, Mul};

use ryft_xla_sys::bindings::{
    MlirAffineExpr, mlirAffineAddExprGet, mlirAffineBinaryOpExprGetLHS, mlirAffineBinaryOpExprGetRHS,
    mlirAffineCeilDivExprGet, mlirAffineConstantExprGet, mlirAffineConstantExprGetValue, mlirAffineDimExprGet,
    mlirAffineDimExprGetPosition, mlirAffineExprCompose, mlirAffineExprGetLargestKnownDivisor,
    mlirAffineExprIsAFloorDiv, mlirAffineExprIsFunctionOfDim, mlirAffineExprIsMultipleOf, mlirAffineExprIsPureAffine,
    mlirAffineExprIsSymbolicOrConstant, mlirAffineExprShiftDims, mlirAffineExprShiftSymbols, mlirAffineFloorDivExprGet,
    mlirAffineModExprGet, mlirAffineMulExprGet, mlirAffineSymbolExprGet, mlirAffineSymbolExprGetPosition,
    mlirSimplifyAffineExpr,
};

use crate::{AffineMap, Context, mlir_subtype_trait_impls};

/// [`AffineExpression`]s are used to represent the mathematical functions that define [`AffineMap`]s.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)
/// for more information.
pub trait AffineExpression<'c, 't: 'c>: Sized + PartialEq + Eq + Display {
    /// Constructs a new instance of this type from the provided handle that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn from_c_api(handle: MlirAffineExpr, context: &'c Context<'t>) -> Option<Self>;

    /// Returns the [`MlirAffineExpr`] that corresponds to this instance and which can be passed to functions
    /// in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn to_c_api(&self) -> MlirAffineExpr;

    /// Returns a reference to the [`Context`] that owns this affine expression.
    fn context(&self) -> &'c Context<'t>;

    /// Returns `true` if this type is an instance of `A`.
    fn is<A: AffineExpression<'c, 't>>(&self) -> bool {
        Self::cast::<A>(&self).is_some()
    }

    /// Tries to cast this type to an instance of `A` (e.g., an instance of [`ConstantAffineExpressionRef`]).
    /// If this is not an instance of the specified type, this function will return [`None`].
    fn cast<A: AffineExpression<'c, 't>>(&self) -> Option<A> {
        unsafe { A::from_c_api(self.to_c_api(), self.context()) }
    }

    /// Up-casts this affine expression to an instance of [`AffineExpression`].
    fn as_ref(&self) -> AffineExpressionRef<'c, 't> {
        unsafe { AffineExpressionRef::from_c_api(self.to_c_api(), self.context()).unwrap() }
    }

    /// Returns `true` if this expression is made out of only symbols and constants (i.e., it does not involve any
    /// dimensional identifiers).
    fn is_symbolic_or_constant(&self) -> bool {
        unsafe { mlirAffineExprIsSymbolicOrConstant(self.to_c_api()) }
    }

    /// Returns `true` if this expression is a pure affine expression (i.e., binary operations are only allowed to
    /// involve constants; no symbols or dimensional identifiers appear in any binary operations).
    fn is_pure_affine(&self) -> bool {
        unsafe { mlirAffineExprIsPureAffine(self.to_c_api()) }
    }

    /// Returns `true` if this expression is a semi-affine expression (i.e., not a pure affine expression). Refer to
    /// [`AffineExpression::is_pure_affine`] for information on what constitutes a pure affine expression.
    fn is_semi_affine(&self) -> bool {
        return !self.is_pure_affine();
    }

    /// Returns the largest known integral divisor of this affine expression.
    fn largest_known_divisor(&self) -> u64 {
        unsafe { mlirAffineExprGetLargestKnownDivisor(self.to_c_api()).cast_unsigned() }
    }

    /// Returns `true` if the value expression evaluates to is a multiple of the provided factor.
    fn is_multiple_of(&self, factor: u64) -> bool {
        unsafe { mlirAffineExprIsMultipleOf(self.to_c_api(), factor.cast_signed()) }
    }

    /// Returns `true` if the expression is a function of the [`DimensionAffineExpressionRef`]
    /// with the provided dimension.
    fn is_function_of_dimension(&self, dimension: usize) -> bool {
        unsafe { mlirAffineExprIsFunctionOfDim(self.to_c_api(), dimension.cast_signed()) }
    }

    /// Returns a new [`AffineExpression`] that is the same as this one but with some of its
    /// [`DimensionAffineExpressionRef`]s shifted by `shift`. Specifically, all [`DimensionAffineExpressionRef`]s
    /// for dimensions in the range `[offset, dimension_count)` are shifted by `shift`.
    fn with_shifted_dimensions(
        &self,
        offset: usize,
        dimension_count: usize,
        shift: usize,
    ) -> AffineExpressionRef<'c, 't> {
        unsafe {
            AffineExpressionRef::from_c_api(
                mlirAffineExprShiftDims(self.to_c_api(), dimension_count as u32, shift as u32, offset as u32),
                self.context(),
            )
            .unwrap()
        }
    }

    /// Returns a new [`AffineExpression`] that is the same as this one but with some of its
    /// [`SymbolAffineExpressionRef`]s shifted by `shift`. Specifically, all [`SymbolAffineExpressionRef`]s for symbols
    /// in the range `[offset, symbol_count)` are shifted by `shift`.
    fn with_shifted_symbols(&self, offset: usize, symbol_count: usize, shift: usize) -> AffineExpressionRef<'c, 't> {
        unsafe {
            AffineExpressionRef::from_c_api(
                mlirAffineExprShiftSymbols(self.to_c_api(), symbol_count as u32, shift as u32, offset as u32),
                self.context(),
            )
            .unwrap()
        }
    }

    /// Returns a new [`ModAffineExpressionRef`]that represents the application of the modulus operator
    /// on the provided [`AffineExpression`]s.
    fn r#mod<Rhs: AffineExpression<'c, 't>>(&self, rhs: Rhs) -> ModAffineExpressionRef<'c, 't> {
        ModAffineExpressionRef::new(self.as_ref(), rhs.as_ref())
    }

    /// Returns a new [`ModAffineExpressionRef`]that represents the application of the modulus operator
    /// on the provided [`AffineExpression`]s.
    fn modulus<Rhs: AffineExpression<'c, 't>>(&self, rhs: Rhs) -> ModAffineExpressionRef<'c, 't> {
        self.r#mod(rhs)
    }

    /// Returns a new [`FloorDivAffineExpressionRef`] that represents the application of the "floor-division" operator
    /// on the provided [`AffineExpression`]s. The "floor-division" operator rounds down the result of the division
    /// to the nearest integer.
    fn floor_div<Rhs: AffineExpression<'c, 't>>(&self, rhs: Rhs) -> FloorDivAffineExpressionRef<'c, 't> {
        FloorDivAffineExpressionRef::new(self.as_ref(), rhs.as_ref())
    }

    /// Returns a new [`CeilDivAffineExpressionRef`] that represents the application of the "ceil-division" operator
    /// on the provided [`AffineExpression`]s. The "ceil-division" operator rounds up the result of the division
    /// to the nearest integer.
    fn ceil_div<Rhs: AffineExpression<'c, 't>>(&self, rhs: Rhs) -> CeilDivAffineExpressionRef<'c, 't> {
        CeilDivAffineExpressionRef::new(self.as_ref(), rhs.as_ref())
    }

    /// Composes this affine expression with the provided [`AffineMap`], returning the resulting [`AffineExpression`].
    ///
    /// This function requires that this expression is composable with the provided map and that the number of
    /// [`DimensionAffineExpressionRef`]s nested inside it is smaller than the number of results of the provided
    /// [`AffineMap`]. If a result of the provided map does not have a corresponding [`DimensionAffineExpressionRef`]
    /// nested in this expression, that result will not appear in the resulting [`AffineExpression`].
    ///
    /// # Example
    ///
    /// The following example uses the [`Display`] renderings of [`AffineExpression`] and [`AffineMap`] for convenience:
    ///
    /// ```text
    ///   Affine Expression: `d0 + d2`
    ///          Affine Map: `(d0, d1, d2)[s0, s1] -> (d0 + s1, d1 + s0, d0 + d1 + d2)`
    /// Composed Expression: `d0 * 2 + d1 + d2 + s1`
    /// ```
    fn compose(&self, map: AffineMap<'c, 't>) -> AffineExpressionRef<'c, 't> {
        unsafe {
            AffineExpressionRef::from_c_api(mlirAffineExprCompose(self.to_c_api(), map.to_c_api()), self.context())
                .unwrap()
        }
    }

    /// Simplifies this affine expression by flattening it and performing some amount of simple analysis, returning the
    /// resulting [`AffineExpression`]. This operation has complexity linear in the number of nodes in this expression. If
    /// this expression is a semi-affine expression, then a simplified semi-affine expression will be constructed with
    /// a sorted list of dimensions and symbols.
    fn simplify(&self, dimension_count: usize, symbol_count: usize) -> AffineExpressionRef<'c, 't> {
        unsafe {
            AffineExpressionRef::from_c_api(
                mlirSimplifyAffineExpr(self.to_c_api(), dimension_count as u32, symbol_count as u32),
                self.context(),
            )
            .unwrap()
        }
    }
}

/// Internal helper macro for generating [`Add`] and [`Mul`] implementations for [`AffineExpression`] subtypes.
macro_rules! mlir_affine_expression_operator_impls {
    ($ty:ident) => {
        impl<'c, 't, A: AffineExpression<'c, 't>> Add<A> for $ty<'c, 't> {
            type Output = AddAffineExpressionRef<'c, 't>;

            fn add(self, rhs: A) -> Self::Output {
                AddAffineExpressionRef::new(self, rhs)
            }
        }

        impl<'c, 't, A: AffineExpression<'c, 't>> Mul<A> for $ty<'c, 't> {
            type Output = MulAffineExpressionRef<'c, 't>;

            fn mul(self, rhs: A) -> Self::Output {
                MulAffineExpressionRef::new(self, rhs.as_ref())
            }
        }
    };
}

/// Reference to an MLIR [`AffineExpression`] that is owned by a [`Context`].
#[derive(Copy, Clone)]
pub struct AffineExpressionRef<'c, 't> {
    /// Handle that represents this [`AffineExpression`] in the MLIR C API.
    handle: MlirAffineExpr,

    /// [`Context`] that owns this [`AffineExpression`].
    context: &'c Context<'t>,
}

impl<'c, 't> AffineExpression<'c, 't> for AffineExpressionRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAffineExpr, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context }) }
    }

    unsafe fn to_c_api(&self) -> MlirAffineExpr {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

mlir_subtype_trait_impls!(AffineExpressionRef<'c, 't> as AffineExpression, mlir_type = AffineExpr);

mlir_affine_expression_operator_impls!(AffineExpressionRef);

/// [`AffineExpression`] that represents a specific dimension (identified by its position).
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)
/// for more information.
#[derive(Copy, Clone)]
pub struct DimensionAffineExpressionRef<'c, 't> {
    /// Handle that represents this [`AffineExpression`] in the MLIR C API.
    handle: MlirAffineExpr,

    /// [`Context`] that owns this [`AffineExpression`].
    context: &'c Context<'t>,
}

impl DimensionAffineExpressionRef<'_, '_> {
    /// Returns the position of this [`DimensionAffineExpressionRef`].
    pub fn position(&self) -> usize {
        unsafe { mlirAffineDimExprGetPosition(self.handle).cast_unsigned() }
    }
}

mlir_subtype_trait_impls!(
    DimensionAffineExpressionRef<'c, 't> as AffineExpression,
    mlir_type = AffineExpr,
    mlir_subtype = Dim,
);

mlir_affine_expression_operator_impls!(DimensionAffineExpressionRef);

impl<'t> Context<'t> {
    /// Creates a new [`DimensionAffineExpressionRef`] with the specified position.
    pub fn dimension_affine_expression<'c>(&'c self, position: usize) -> DimensionAffineExpressionRef<'c, 't> {
        unsafe {
            DimensionAffineExpressionRef::from_c_api(
                mlirAffineDimExprGet(*self.handle.borrow_mut(), position.cast_signed()),
                &self,
            )
            .unwrap()
        }
    }
}

/// [`AffineExpression`] that represents a specific symbol (identified by its position).
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)
/// for more information.
#[derive(Copy, Clone)]
pub struct SymbolAffineExpressionRef<'c, 't> {
    /// Handle that represents this [`AffineExpression`] in the MLIR C API.
    handle: MlirAffineExpr,

    /// [`Context`] that owns this [`AffineExpression`].
    context: &'c Context<'t>,
}

impl SymbolAffineExpressionRef<'_, '_> {
    /// Returns the position of this [`SymbolAffineExpressionRef`].
    pub fn position(&self) -> usize {
        unsafe { mlirAffineSymbolExprGetPosition(self.handle).cast_unsigned() }
    }
}

mlir_subtype_trait_impls!(
    SymbolAffineExpressionRef<'c, 't> as AffineExpression,
    mlir_type = AffineExpr,
    mlir_subtype = Symbol,
);

mlir_affine_expression_operator_impls!(SymbolAffineExpressionRef);

impl<'t> Context<'t> {
    /// Creates a new [`SymbolAffineExpressionRef`] with the specified position.
    pub fn symbol_affine_expression<'c>(&'c self, position: usize) -> SymbolAffineExpressionRef<'c, 't> {
        unsafe {
            SymbolAffineExpressionRef::from_c_api(
                mlirAffineSymbolExprGet(*self.handle.borrow_mut(), position.cast_signed()),
                &self,
            )
            .unwrap()
        }
    }
}

/// [`AffineExpression`] that represents a constant `i64` value.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)
/// for more information.
#[derive(Copy, Clone)]
pub struct ConstantAffineExpressionRef<'c, 't> {
    /// Handle that represents this [`AffineExpression`] in the MLIR C API.
    handle: MlirAffineExpr,

    /// [`Context`] that owns this [`AffineExpression`].
    context: &'c Context<'t>,
}

impl ConstantAffineExpressionRef<'_, '_> {
    /// Returns the value of this [`ConstantAffineExpressionRef`].
    pub fn value(&self) -> i64 {
        unsafe { mlirAffineConstantExprGetValue(self.handle) }
    }
}

mlir_subtype_trait_impls!(
    ConstantAffineExpressionRef<'c, 't> as AffineExpression,
    mlir_type = AffineExpr,
    mlir_subtype = Constant,
);

mlir_affine_expression_operator_impls!(ConstantAffineExpressionRef);

impl<'t> Context<'t> {
    /// Creates a new [`ConstantAffineExpressionRef`] with the specified value.
    pub fn constant_affine_expression<'c>(&'c self, value: i64) -> ConstantAffineExpressionRef<'c, 't> {
        unsafe {
            ConstantAffineExpressionRef::from_c_api(mlirAffineConstantExprGet(*self.handle.borrow_mut(), value), &self)
                .unwrap()
        }
    }
}

/// [`AffineExpression`] that represents a binary operation between two [`AffineExpression`]s. This is a "super-type" of
/// [`AddAffineExpressionRef`], [`MulAffineExpressionRef`], [`ModAffineExpressionRef`], [`FloorDivAffineExpressionRef`],
/// and [`CeilDivAffineExpressionRef`].
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)
/// for more information.
#[derive(Copy, Clone)]
pub struct BinaryOperationAffineExpressionRef<'c, 't> {
    /// Handle that represents this [`AffineExpression`] in the MLIR C API.
    handle: MlirAffineExpr,

    /// [`Context`] that owns this [`AffineExpression`].
    context: &'c Context<'t>,
}

impl<'c, 't> BinaryOperationAffineExpressionRef<'c, 't> {
    /// Returns the left-hand side (LHS) operand of this [`BinaryOperationAffineExpressionRef`].
    pub fn lhs_operand(&self) -> AffineExpressionRef<'c, 't> {
        unsafe { AffineExpressionRef::from_c_api(mlirAffineBinaryOpExprGetLHS(self.handle), self.context).unwrap() }
    }

    /// Returns the right-hand side (RHS) operand of this [`BinaryOperationAffineExpressionRef`].
    pub fn rhs_operand(&self) -> AffineExpressionRef<'c, 't> {
        unsafe { AffineExpressionRef::from_c_api(mlirAffineBinaryOpExprGetRHS(self.handle), self.context).unwrap() }
    }
}

mlir_subtype_trait_impls!(
    BinaryOperationAffineExpressionRef<'c, 't> as AffineExpression,
    mlir_type = AffineExpr,
    mlir_subtype = Binary,
);

mlir_affine_expression_operator_impls!(BinaryOperationAffineExpressionRef);

/// [`AffineExpression`] that represents the addition of two [`AffineExpression`]s.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)
/// for more information.
#[derive(Copy, Clone)]
pub struct AddAffineExpressionRef<'c, 't> {
    /// Handle that represents this [`AffineExpression`] in the MLIR C API.
    handle: MlirAffineExpr,

    /// [`Context`] that owns this [`AffineExpression`].
    context: &'c Context<'t>,
}

impl<'c, 't> AddAffineExpressionRef<'c, 't> {
    /// Creates a new [`AddAffineExpressionRef`] using the provided [`AffineExpression`]s as its operands.
    pub fn new<LHS: AffineExpression<'c, 't>, RHS: AffineExpression<'c, 't>>(lhs: LHS, rhs: RHS) -> Self {
        unsafe {
            AddAffineExpressionRef::from_c_api(mlirAffineAddExprGet(lhs.to_c_api(), rhs.to_c_api()), lhs.context())
                .unwrap()
        }
    }
}

mlir_subtype_trait_impls!(
    AddAffineExpressionRef<'c, 't> as AffineExpression,
    mlir_type = AffineExpr,
    mlir_subtype = Add,
);

mlir_affine_expression_operator_impls!(AddAffineExpressionRef);

/// [`AffineExpression`] that represents the multiplication of two [`AffineExpression`]s.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)
/// for more information.
#[derive(Copy, Clone)]
pub struct MulAffineExpressionRef<'c, 't> {
    /// Handle that represents this [`AffineExpression`] in the MLIR C API.
    handle: MlirAffineExpr,

    /// [`Context`] that owns this [`AffineExpression`].
    context: &'c Context<'t>,
}

impl<'c, 't> MulAffineExpressionRef<'c, 't> {
    /// Creates a new [`MulAffineExpressionRef`] using the provided [`AffineExpression`]s as its operands.
    pub fn new<LHS: AffineExpression<'c, 't>, RHS: AffineExpression<'c, 't>>(lhs: LHS, rhs: RHS) -> Self {
        unsafe {
            MulAffineExpressionRef::from_c_api(mlirAffineMulExprGet(lhs.to_c_api(), rhs.to_c_api()), lhs.context())
                .unwrap()
        }
    }
}

mlir_subtype_trait_impls!(
    MulAffineExpressionRef<'c, 't> as AffineExpression,
    mlir_type = AffineExpr,
    mlir_subtype = Mul,
);

mlir_affine_expression_operator_impls!(MulAffineExpressionRef);

/// [`AffineExpression`] that represents the application of the modulus operator on two [`AffineExpression`]s.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)
/// for more information.
#[derive(Copy, Clone)]
pub struct ModAffineExpressionRef<'c, 't> {
    /// Handle that represents this [`AffineExpression`] in the MLIR C API.
    handle: MlirAffineExpr,

    /// [`Context`] that owns this [`AffineExpression`].
    context: &'c Context<'t>,
}

impl<'c, 't> ModAffineExpressionRef<'c, 't> {
    /// Creates a new [`ModAffineExpressionRef`]using the provided [`AffineExpression`]s as its operands.
    pub fn new<LHS: AffineExpression<'c, 't>, RHS: AffineExpression<'c, 't>>(lhs: LHS, rhs: RHS) -> Self {
        unsafe {
            ModAffineExpressionRef::from_c_api(mlirAffineModExprGet(lhs.to_c_api(), rhs.to_c_api()), lhs.context())
                .unwrap()
        }
    }
}

mlir_subtype_trait_impls!(
    ModAffineExpressionRef<'c, 't> as AffineExpression,
    mlir_type = AffineExpr,
    mlir_subtype = Mod,
);

mlir_affine_expression_operator_impls!(ModAffineExpressionRef);

/// [`AffineExpression`] that represents the application of the "floor-division" operator on two [`AffineExpression`]s.
/// The "floor-division" operator rounds down the result of the division to the nearest integer.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)
/// for more information.
#[derive(Copy, Clone)]
pub struct FloorDivAffineExpressionRef<'c, 't> {
    /// Handle that represents this [`AffineExpression`] in the MLIR C API.
    handle: MlirAffineExpr,

    /// [`Context`] that owns this [`AffineExpression`].
    context: &'c Context<'t>,
}

impl<'c, 't> FloorDivAffineExpressionRef<'c, 't> {
    /// Creates a new [`FloorDivAffineExpressionRef`] using the provided [`AffineExpression`]s as its operands.
    pub fn new<LHS: AffineExpression<'c, 't>, RHS: AffineExpression<'c, 't>>(lhs: LHS, rhs: RHS) -> Self {
        unsafe {
            FloorDivAffineExpressionRef::from_c_api(
                mlirAffineFloorDivExprGet(lhs.to_c_api(), rhs.to_c_api()),
                lhs.context(),
            )
            .unwrap()
        }
    }
}

impl<'c, 't> AffineExpression<'c, 't> for FloorDivAffineExpressionRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAffineExpr, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirAffineExprIsAFloorDiv(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAffineExpr {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

mlir_subtype_trait_impls!(FloorDivAffineExpressionRef<'c, 't> as AffineExpression, mlir_type = AffineExpr);

mlir_affine_expression_operator_impls!(FloorDivAffineExpressionRef);

/// [`AffineExpression`] that represents the application of the "ceil-division" operator on two [`AffineExpression`]s.
/// The "ceil-division" operator rounds up the result of the division to the nearest integer.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)
/// for more information.
#[derive(Copy, Clone)]
pub struct CeilDivAffineExpressionRef<'c, 't> {
    /// Handle that represents this [`AffineExpression`] in the MLIR C API.
    handle: MlirAffineExpr,

    /// [`Context`] that owns this [`AffineExpression`].
    context: &'c Context<'t>,
}

impl<'c, 't> CeilDivAffineExpressionRef<'c, 't> {
    /// Creates a new [`CeilDivAffineExpressionRef`] using the provided [`AffineExpression`]s as its operands.
    pub fn new<LHS: AffineExpression<'c, 't>, RHS: AffineExpression<'c, 't>>(lhs: LHS, rhs: RHS) -> Self {
        unsafe {
            CeilDivAffineExpressionRef::from_c_api(
                mlirAffineCeilDivExprGet(lhs.to_c_api(), rhs.to_c_api()),
                lhs.context(),
            )
            .unwrap()
        }
    }
}

mlir_subtype_trait_impls!(
    CeilDivAffineExpressionRef<'c, 't> as AffineExpression,
    mlir_type = AffineExpr,
    mlir_subtype = CeilDiv,
);

mlir_affine_expression_operator_impls!(CeilDivAffineExpressionRef);

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Context;

    use super::*;

    #[test]
    fn test_dimension_affine_expression() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let dimension_10 = context.dimension_affine_expression(10);
        assert_eq!(&context, dimension_0.context());
        assert_eq!(dimension_0.position(), 0);
        assert_eq!(dimension_1.position(), 1);
        assert_eq!(dimension_10.position(), 10);
        assert_eq!(dimension_0.to_string(), "d0");
        assert_eq!(dimension_1.to_string(), "d1");
        assert_eq!(dimension_10.to_string(), "d10");
    }

    #[test]
    fn test_dimension_affine_expression_equality() {
        let context = Context::new();
        let dimensions_0_0 = context.dimension_affine_expression(0);
        let dimension_0_1 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);

        // Same dimensions from the same context must be equal because they are "uniqued".
        assert_eq!(dimensions_0_0, dimension_0_1);
        assert_ne!(dimensions_0_0, dimension_1);

        // Same dimensions from different contexts must not be equal.
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        assert_ne!(dimensions_0_0, dimension_0);
    }

    #[test]
    fn test_symbol_affine_expression() {
        let context = Context::new();
        let symbol_0 = context.symbol_affine_expression(0);
        let symbol_1 = context.symbol_affine_expression(1);
        let symbol_2 = context.symbol_affine_expression(10);
        assert_eq!(&context, symbol_0.context());
        assert_eq!(symbol_0.position(), 0);
        assert_eq!(symbol_1.position(), 1);
        assert_eq!(symbol_2.position(), 10);
        assert_eq!(symbol_0.to_string(), "s0");
        assert_eq!(symbol_1.to_string(), "s1");
        assert_eq!(symbol_2.to_string(), "s10");
    }

    #[test]
    fn test_symbol_affine_expression_equality() {
        let context = Context::new();
        let symbol_0_0 = context.symbol_affine_expression(0);
        let symbol_0_1 = context.symbol_affine_expression(0);
        let symbol_1 = context.symbol_affine_expression(1);

        // Same symbols from the same context must be equal because they are "uniqued".
        assert_eq!(symbol_0_0, symbol_0_1);
        assert_ne!(symbol_0_0, symbol_1);

        // Same symbols from different contexts must not be equal.
        let context = Context::new();
        let symbol_2 = context.symbol_affine_expression(0);
        assert_ne!(symbol_0_0, symbol_2);
    }

    #[test]
    fn test_constant_affine_expression() {
        let context = Context::new();
        let constant_0 = context.constant_affine_expression(0);
        let constant_1 = context.constant_affine_expression(42);
        let constant_2 = context.constant_affine_expression(-5);
        assert_eq!(&context, constant_0.context());
        assert_eq!(constant_0.value(), 0);
        assert_eq!(constant_1.value(), 42);
        assert_eq!(constant_2.value(), -5);
        assert_eq!(constant_0.to_string(), "0");
        assert_eq!(constant_1.to_string(), "42");
        assert_eq!(constant_2.to_string(), "-5");
    }

    #[test]
    fn test_constant_affine_expression_equality() {
        let context = Context::new();
        let constant_0 = context.constant_affine_expression(42);
        let constant_1 = context.constant_affine_expression(42);
        let constant_2 = context.constant_affine_expression(43);

        // Same constants from the same context must be equal because they are "uniqued".
        assert_eq!(constant_0, constant_1);
        assert_ne!(constant_0, constant_2);

        // Same constants from different contexts must not be equal.
        let context = Context::new();
        let constant_3 = context.constant_affine_expression(42);
        assert_ne!(constant_0, constant_3);
    }

    #[test]
    fn test_add_affine_expression() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0).as_ref();
        let dimension_1 = context.dimension_affine_expression(1).as_ref();
        let constant_0 = context.constant_affine_expression(5).as_ref();

        let add_expression_0 = dimension_0 + dimension_1;
        assert_eq!(&context, add_expression_0.context());
        assert_eq!(add_expression_0.to_string(), "d0 + d1");

        let add_expression_1 = dimension_0 + constant_0;
        assert_eq!(add_expression_1.to_string(), "d0 + 5");

        let binary_expression = add_expression_0.cast::<BinaryOperationAffineExpressionRef>().unwrap();
        assert_eq!(binary_expression.lhs_operand(), dimension_0);
        assert_eq!(binary_expression.rhs_operand(), dimension_1);
    }

    #[test]
    fn test_mul_affine_expression() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0).as_ref();
        let constant_0 = context.constant_affine_expression(3).as_ref();

        let mul_expression = dimension_0 * constant_0;
        assert_eq!(&context, mul_expression.context());
        assert_eq!(mul_expression.to_string(), "d0 * 3");

        let binary_expression = mul_expression.cast::<BinaryOperationAffineExpressionRef>().unwrap();
        assert_eq!(binary_expression.lhs_operand(), dimension_0);
        assert_eq!(binary_expression.rhs_operand(), constant_0);
    }

    #[test]
    fn test_mod_affine_expression() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let constant_0 = context.constant_affine_expression(8);

        let mod_expression_0 = dimension_0.r#mod(constant_0);
        assert_eq!(&context, mod_expression_0.context());
        assert_eq!(mod_expression_0.to_string(), "d0 mod 8");

        let mod_expression_1 = dimension_0.modulus(constant_0);
        assert_eq!(mod_expression_0, mod_expression_1);

        let binary_expression = mod_expression_0.cast::<BinaryOperationAffineExpressionRef>().unwrap();
        assert_eq!(binary_expression.lhs_operand(), dimension_0);
        assert_eq!(binary_expression.rhs_operand(), constant_0);
    }

    #[test]
    fn test_floor_div_affine_expression() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let constant_0 = context.constant_affine_expression(4);

        let floor_div_expression = dimension_0.floor_div(constant_0);
        assert_eq!(&context, floor_div_expression.context());
        assert_eq!(floor_div_expression.to_string(), "d0 floordiv 4");

        let binary_expression = floor_div_expression.cast::<BinaryOperationAffineExpressionRef>().unwrap();
        assert_eq!(binary_expression.lhs_operand(), dimension_0);
        assert_eq!(binary_expression.rhs_operand(), constant_0);
    }

    #[test]
    fn test_ceil_div_affine_expression() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let constant_0 = context.constant_affine_expression(4);

        let ceil_div_expression = dimension_0.ceil_div(constant_0);
        assert_eq!(&context, ceil_div_expression.context());
        assert_eq!(ceil_div_expression.to_string(), "d0 ceildiv 4");

        let binary_expression = ceil_div_expression.cast::<BinaryOperationAffineExpressionRef>().unwrap();
        assert_eq!(binary_expression.lhs_operand(), dimension_0);
        assert_eq!(binary_expression.rhs_operand(), constant_0);
    }

    #[test]
    fn test_complex_affine_expression() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let constant_0 = context.constant_affine_expression(2);
        let constant_1 = context.constant_affine_expression(3);
        let expression = ((dimension_0 * constant_0) + dimension_1) * constant_1;
        assert_eq!(expression.to_string(), "(d0 * 2 + d1) * 3");
    }

    #[test]
    fn test_affine_expression_casting() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_0_erased = dimension_0.as_ref();
        let dimension_0_erased_casted = dimension_0_erased.cast::<DimensionAffineExpressionRef>().unwrap();
        let constant_1 = context.constant_affine_expression(42);
        let symbol_0 = context.symbol_affine_expression(0);
        assert!(dimension_0.is::<AffineExpressionRef>());
        assert!(dimension_0.is::<DimensionAffineExpressionRef>());
        assert!(!dimension_0.is::<SymbolAffineExpressionRef>());
        assert!(!dimension_0.is::<ConstantAffineExpressionRef>());
        assert!(!dimension_0.is::<FloorDivAffineExpressionRef>());
        assert!(symbol_0.is::<AffineExpressionRef>());
        assert!(!symbol_0.is::<DimensionAffineExpressionRef>());
        assert!(symbol_0.is::<SymbolAffineExpressionRef>());
        assert!(!symbol_0.is::<ConstantAffineExpressionRef>());
        assert!(!symbol_0.is::<FloorDivAffineExpressionRef>());
        assert!(constant_1.is::<AffineExpressionRef>());
        assert!(!constant_1.is::<DimensionAffineExpressionRef>());
        assert!(!constant_1.is::<SymbolAffineExpressionRef>());
        assert!(constant_1.is::<ConstantAffineExpressionRef>());
        assert!(!constant_1.is::<FloorDivAffineExpressionRef>());
        assert_eq!(dimension_0_erased_casted, dimension_0);
        assert!(!symbol_0.is::<DimensionAffineExpressionRef>());
        let bad_handle = MlirAffineExpr { ptr: std::ptr::null() };
        assert!(unsafe { AffineExpressionRef::from_c_api(bad_handle, &context).is_none() });
    }

    #[test]
    fn test_affine_expression_is_symbolic_or_constant() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.constant_affine_expression(42);
        let symbol_0 = context.symbol_affine_expression(0);
        assert!(!dimension_0.is_symbolic_or_constant());
        assert!(symbol_0.is_symbolic_or_constant());
        assert!(dimension_1.is_symbolic_or_constant());
        assert!(!(dimension_0 + symbol_0).is_symbolic_or_constant());
        assert!((symbol_0 + dimension_1).is_symbolic_or_constant());
    }

    #[test]
    fn test_affine_expression_is_pure_affine_and_semi_affine() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let constant_0 = context.constant_affine_expression(2);

        let pure_expression = dimension_0 + dimension_1;
        assert!(pure_expression.is_pure_affine());
        assert!(!pure_expression.is_semi_affine());

        let pure_expression = dimension_0 * constant_0;
        assert!(pure_expression.is_pure_affine());

        let semi_affine_expression = dimension_0 * dimension_1;
        assert!(!semi_affine_expression.is_pure_affine());
        assert!(semi_affine_expression.is_semi_affine());
    }

    #[test]
    fn test_affine_expression_largest_known_divisor() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let constant_0 = context.constant_affine_expression(4);
        let constant_1 = context.constant_affine_expression(8);
        assert_eq!((dimension_0 * constant_1).largest_known_divisor(), 8);
        assert_eq!((dimension_0 * constant_0).largest_known_divisor(), 4);
    }

    #[test]
    fn test_affine_expression_is_multiple_of() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let constant_0 = context.constant_affine_expression(8);
        let expression = dimension_0 * constant_0;
        assert!(expression.is_multiple_of(2));
        assert!(expression.is_multiple_of(4));
        assert!(expression.is_multiple_of(8));
        assert!(!expression.is_multiple_of(16));
    }

    #[test]
    fn test_affine_expression_is_function_of_dimension() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let expression = dimension_0 + dimension_1;
        let symbol_0 = context.symbol_affine_expression(0);
        assert!(dimension_0.is_function_of_dimension(0));
        assert!(!dimension_0.is_function_of_dimension(1));
        assert!(!dimension_1.is_function_of_dimension(0));
        assert!(dimension_1.is_function_of_dimension(1));
        assert!(!symbol_0.is_function_of_dimension(0));
        assert!(expression.is_function_of_dimension(0));
        assert!(expression.is_function_of_dimension(1));
    }

    #[test]
    fn test_affine_expression_with_shifted_dimensions() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let dimension_2 = context.dimension_affine_expression(2);
        let expression = (dimension_0 + dimension_1) + dimension_2;
        let shifted_expression = expression.with_shifted_dimensions(1, 3, 2);
        assert_eq!(shifted_expression.to_string(), "d0 + d3 + d4");
    }

    #[test]
    fn test_affine_expression_with_shifted_symbols() {
        let context = Context::new();
        let symbol_0 = context.symbol_affine_expression(0);
        let symbol_1 = context.symbol_affine_expression(1);
        let symbol_2 = context.symbol_affine_expression(2);
        let expression = (symbol_0 + symbol_1) + symbol_2;
        let shifted_expression = expression.with_shifted_symbols(1, 3, 2);
        assert_eq!(shifted_expression.to_string(), "s0 + s3 + s4");
    }

    #[test]
    fn test_affine_expression_compose() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let constant_0 = context.constant_affine_expression(2);
        let constant_1 = context.constant_affine_expression(3);
        let expression_0 = dimension_0 + dimension_1;
        let expression_1 = dimension_0 * constant_0;
        let expression_2 = dimension_1 * constant_1;
        let map = context.affine_map(2, 0, &[expression_1, expression_2]);
        let composed = expression_0.compose(map);
        assert_eq!(composed.to_string(), "d0 * 2 + d1 * 3");
    }

    #[test]
    fn test_affine_expression_simplify() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let constant_0 = context.constant_affine_expression(2);
        let simplified_expression_0 = ((dimension_0 * constant_0) + dimension_1).simplify(2, 0);
        let simplified_expression_1 = (dimension_0 + dimension_1).simplify(2, 0);
        assert_eq!(simplified_expression_0.to_string(), "d0 * 2 + d1");
        assert_eq!(simplified_expression_1.to_string(), "d0 + d1");
    }
}
