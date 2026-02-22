use std::fmt::{Debug, Display};

use ryft_xla_sys::bindings::{
    MlirIntegerSet, mlirIntegerSetDump, mlirIntegerSetEmptyGet, mlirIntegerSetEqual, mlirIntegerSetGet,
    mlirIntegerSetGetConstraint, mlirIntegerSetGetNumConstraints, mlirIntegerSetGetNumDims,
    mlirIntegerSetGetNumEqualities, mlirIntegerSetGetNumInequalities, mlirIntegerSetGetNumInputs,
    mlirIntegerSetGetNumSymbols, mlirIntegerSetIsCanonicalEmpty, mlirIntegerSetIsConstraintEq, mlirIntegerSetPrint,
    mlirIntegerSetReplaceGet,
};

use crate::{AffineExpression, AffineExpressionRef, Context, support::write_to_formatter_callback};

/// Constraint for an [`IntegerSet`].
pub struct IntegerSetConstraint<'c, 't> {
    /// Affine expression over the set of dimensions and symbols that are involved in the integer set.
    expression: AffineExpressionRef<'c, 't>,

    /// Boolean flag indicating whether the constraint is an equality or an inequality constraint.
    is_equality: bool,
}

/// Represents an MLIR integer set which is a set of points from the integer lattice constrained by affine
/// equality/inequality constraints (i.e., [`IntegerSetConstraint`]s). Such sets are meant to represent integer sets
/// in the IR (e.g., for `affine.if` operations and as attributes of other operations). They are typically expected
/// to contain only a handful of affine constraints, and are immutable like [`AffineMap`](crate::AffineMap)s. Integer
/// sets are not "unique'd" internally by their owning [`Context`] unless the number of constraints they contain is
/// below a certain threshold (although the [`AffineExpression`] that make up their equality and inequality constraints
/// are themselves "unique'd").
pub struct IntegerSet<'c, 't> {
    /// Handle that represents this [`IntegerSet`] in the MLIR C API.
    handle: MlirIntegerSet,

    /// [`Context`] that owns this [`IntegerSet`].
    context: &'c Context<'t>,
}

impl<'c, 't> IntegerSet<'c, 't> {
    /// Constructs a new [`IntegerSet`] from the provided [`MlirIntegerSet`] handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirIntegerSet, context: &'c Context<'t>) -> Self {
        Self { handle, context }
    }

    /// Returns the [`MlirIntegerSet`] that corresponds to this [`IntegerSet`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirIntegerSet {
        self.handle
    }

    /// Returns a reference to the [`Context`] that owns this [`IntegerSet`].
    pub fn context(&self) -> &'c Context<'t> {
        self.context
    }

    /// Returns the number of dimensions of this [`IntegerSet`].
    pub fn dimension_count(&self) -> usize {
        unsafe { mlirIntegerSetGetNumDims(self.handle).cast_unsigned() }
    }

    /// Returns the number of symbols of this [`IntegerSet`].
    pub fn symbol_count(&self) -> usize {
        unsafe { mlirIntegerSetGetNumSymbols(self.handle).cast_unsigned() }
    }

    /// Returns the number of inputs (i.e., number of dimensions and symbols combined) of this [`IntegerSet`].
    pub fn input_count(&self) -> usize {
        unsafe { mlirIntegerSetGetNumInputs(self.handle).cast_unsigned() }
    }

    /// Returns the number of equality constraints of this [`IntegerSet`].
    pub fn equality_constraint_count(&self) -> usize {
        unsafe { mlirIntegerSetGetNumEqualities(self.handle).cast_unsigned() }
    }

    /// Returns the number of inequality constraints of this [`IntegerSet`].
    pub fn inequality_constraint_count(&self) -> usize {
        unsafe { mlirIntegerSetGetNumInequalities(self.handle).cast_unsigned() }
    }

    /// Returns the number of (both equality and inequality) constraints of this [`IntegerSet`].
    pub fn constraint_count(&self) -> usize {
        unsafe { mlirIntegerSetGetNumConstraints(self.handle).cast_unsigned() }
    }

    /// Returns the constraints of this [`IntegerSet`].
    pub fn constraints(&self) -> impl Iterator<Item = IntegerSetConstraint<'c, 't>> {
        (0..self.constraint_count()).map(|index| self.constraint(index).unwrap())
    }

    /// Returns the `index`-th constraint of this [`IntegerSet`] and [`None`] if the provided index is out of bounds.
    pub fn constraint(&self, index: usize) -> Option<IntegerSetConstraint<'c, 't>> {
        if index >= self.constraint_count() {
            None
        } else {
            unsafe {
                let constraint_handle = mlirIntegerSetGetConstraint(self.handle, index.cast_signed());
                if constraint_handle.ptr.is_null() {
                    None
                } else {
                    Some(IntegerSetConstraint {
                        expression: AffineExpressionRef::from_c_api(constraint_handle, self.context).unwrap(),
                        is_equality: mlirIntegerSetIsConstraintEq(self.handle, index.cast_signed()),
                    })
                }
            }
        }
    }

    /// Returns `true` if this [`IntegerSet`] is a canonical empty set. Refer to [`Context::empty_integer_set`]
    /// for more information.
    pub fn is_empty(&self) -> bool {
        unsafe { mlirIntegerSetIsCanonicalEmpty(self.handle) }
    }

    /// Returns an [`IntegerSet`] which is the same as this integer set but with its dimensions and symbols replaced
    /// by the provided [`AffineExpression`]s. The provided replacement [`AffineExpression`]s must be at least as many as the
    /// corresponding number of dimensions or symbols in the current set. The resulting integer set will have
    /// `result_dimension_count` dimensions and `result_symbol_count` symbols.
    pub fn replace<D: AffineExpression<'c, 't>, S: AffineExpression<'c, 't>>(
        &self,
        dimension_replacements: &[D],
        symbol_replacements: &[S],
        result_dimension_count: usize,
        result_symbol_count: usize,
    ) -> Self {
        unsafe {
            let dimension_replacements = dimension_replacements.iter().map(|expr| expr.to_c_api()).collect::<Vec<_>>();
            let symbol_replacements = symbol_replacements.iter().map(|expr| expr.to_c_api()).collect::<Vec<_>>();
            Self::from_c_api(
                mlirIntegerSetReplaceGet(
                    self.handle,
                    dimension_replacements.as_ptr() as *const _,
                    symbol_replacements.as_ptr() as *const _,
                    result_dimension_count.cast_signed(),
                    result_symbol_count.cast_signed(),
                ),
                self.context,
            )
        }
    }

    /// Dumps this [`IntegerSet`] to the standard error stream.
    pub fn dump(&self) {
        unsafe { mlirIntegerSetDump(self.handle) }
    }
}

impl PartialEq for IntegerSet<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        // Note that this is a "shallow" comparison of the two integer sets. Only sets with some small number of
        // constraints are uniqued and will be considered equal based on this check. Sets that represent the same
        // integer set but with different constraints may be considered non-equal by this check. Set difference
        // followed by an (expensive) emptiness check should be used to check for equivalence of the underlying
        // integer sets.
        unsafe { mlirIntegerSetEqual(self.handle, other.handle) }
    }
}

impl Eq for IntegerSet<'_, '_> {}

impl Display for IntegerSet<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirIntegerSetPrint(
                self.handle,
                Some(write_to_formatter_callback),
                &mut data as *mut _ as *mut std::ffi::c_void,
            );
        }
        data.1
    }
}

impl Debug for IntegerSet<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "IntegerSet[{self}]")
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`IntegerSet`] with the given number of dimensions and symbols, that is defined by the provided
    /// [`IntegerSetConstraint`]s. The resulting integer set is owned by this [`Context`].
    pub fn integer_set<'c>(
        &'c self,
        dimension_count: usize,
        symbol_count: usize,
        constraints: &[&IntegerSetConstraint<'c, 't>],
    ) -> IntegerSet<'c, 't> {
        unsafe {
            let constraint_expressions = constraints.iter().map(|c| c.expression.to_c_api()).collect::<Vec<_>>();
            let constraint_equalities = constraints.iter().map(|c| c.is_equality).collect::<Vec<_>>();
            IntegerSet::from_c_api(
                mlirIntegerSetGet(
                    *self.handle.borrow_mut(),
                    dimension_count.cast_signed(),
                    symbol_count.cast_signed(),
                    constraints.len().cast_signed(),
                    constraint_expressions.as_ptr() as *const _,
                    constraint_equalities.as_ptr(),
                ),
                self,
            )
        }
    }

    /// Creates a canonically empty [`IntegerSet`] with the given number of dimensions and symbols.
    /// The resulting integer set is owned by this [`Context`].
    pub fn empty_integer_set<'c>(&'c self, dimension_count: usize, symbol_count: usize) -> IntegerSet<'c, 't> {
        unsafe {
            IntegerSet::from_c_api(
                mlirIntegerSetEmptyGet(
                    *self.handle.borrow_mut(),
                    dimension_count.cast_signed(),
                    symbol_count.cast_signed(),
                ),
                self,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Context;

    use super::*;

    #[test]
    fn test_integer_set() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let constant_0 = context.constant_affine_expression(5);
        let constant_1 = context.constant_affine_expression(-10);
        let constraint_0 = IntegerSetConstraint { expression: (dimension_0 + constant_0).as_ref(), is_equality: false };
        let constraint_1 = IntegerSetConstraint { expression: (dimension_0 + constant_1).as_ref(), is_equality: false };
        let set = context.integer_set(1, 0, &[&constraint_0, &constraint_1]);
        assert_eq!(&context, set.context());
        assert_eq!(set.dimension_count(), 1);
        assert_eq!(set.symbol_count(), 0);
        assert_eq!(set.input_count(), 1);
        assert_eq!(set.constraint_count(), 2);
        assert!(set.constraint(2).is_none());
    }

    #[test]
    fn test_integer_set_constraints() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let constant_0 = context.constant_affine_expression(-5);
        let constraint = IntegerSetConstraint { expression: (dimension_0 + constant_0).as_ref(), is_equality: true };
        let set = context.integer_set(1, 0, &[&constraint]);
        assert_eq!(set.equality_constraint_count(), 1);
        assert_eq!(set.inequality_constraint_count(), 0);
        let constraint_0 = set.constraint(0).unwrap();
        assert_eq!(constraint_0.expression, constraint.expression);
        assert_eq!(constraint_0.is_equality, true);
    }

    #[test]
    fn test_integer_set_constraints_iterator() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let constant_0 = context.constant_affine_expression(-5);
        let constant_1 = context.constant_affine_expression(-10);
        let constraint_0 = IntegerSetConstraint { expression: (dimension_0 + constant_0).as_ref(), is_equality: true };
        let constraint_1 = IntegerSetConstraint { expression: (dimension_0 + constant_1).as_ref(), is_equality: false };
        let set = context.integer_set(1, 0, &[&constraint_0, &constraint_1]);
        let constraints = set.constraints().collect::<Vec<_>>();
        assert_eq!(constraints.len(), 2);
        assert_eq!(constraints[0].is_equality, true);
        assert_eq!(constraints[1].is_equality, false);
    }

    #[test]
    fn test_empty_integer_set() {
        let context = Context::new();
        let set = context.empty_integer_set(2, 1);
        assert_eq!(set.dimension_count(), 2);
        assert_eq!(set.symbol_count(), 1);
        assert!(set.is_empty());
        assert_eq!(set.constraint_count(), 1);
    }

    #[test]
    fn test_integer_set_equality() {
        let context = Context::new();
        let dim0 = context.dimension_affine_expression(0);

        let const_neg5 = context.constant_affine_expression(-5);
        let constraint_expr = (dim0.as_ref() + const_neg5.as_ref()).as_ref();
        let constraint = IntegerSetConstraint { expression: constraint_expr.cast().unwrap(), is_equality: true };

        // Same sets from the same context must be equal (if they're uniqued)
        let set1 = context.integer_set(1, 0, &[&constraint]);
        let set2 = context.integer_set(1, 0, &[&constraint]);
        assert_eq!(set1, set2);

        // Same sets from different contexts must not be equal
        let other_context = Context::new();
        let other_dim0 = other_context.dimension_affine_expression(0);
        let other_const_neg5 = other_context.constant_affine_expression(-5);
        let other_constraint_expr = (other_dim0.as_ref() + other_const_neg5.as_ref()).as_ref();
        let other_constraint =
            IntegerSetConstraint { expression: other_constraint_expr.cast().unwrap(), is_equality: true };
        let set3 = other_context.integer_set(1, 0, &[&other_constraint]);
        assert_ne!(set1, set3);
    }

    #[test]
    fn test_integer_set_replace() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let constant_0 = context.constant_affine_expression(-5);
        let constant_1 = context.constant_affine_expression(2);
        let constant_2 = context.constant_affine_expression(3);
        let symbol_0 = context.symbol_affine_expression(0);
        let constraint = IntegerSetConstraint {
            expression: ((dimension_0 + dimension_1) + constant_0).as_ref(),
            is_equality: false,
        };
        let set = context.integer_set(2, 0, &[&constraint]);
        let replaced_set =
            set.replace(&[(dimension_0 * constant_1).as_ref(), (dimension_1 * constant_2).as_ref()], &[symbol_0], 2, 1);
        assert_eq!(replaced_set.dimension_count(), 2);
        assert_eq!(replaced_set.symbol_count(), 1);
        assert_eq!(replaced_set.constraint_count(), 1);
    }

    #[test]
    fn test_integer_set_display_and_debug() {
        let context = Context::new();
        let empty_set = context.empty_integer_set(2, 1);
        assert_eq!(format!("{}", empty_set), "(d0, d1)[s0] : (1 == 0)");
        assert_eq!(format!("{:?}", empty_set), "IntegerSet[(d0, d1)[s0] : (1 == 0)]");
    }

    #[test]
    fn test_integer_set_dump() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let constant_0 = context.constant_affine_expression(5);
        let constant_1 = context.constant_affine_expression(-10);
        let constraint_0 = IntegerSetConstraint { expression: (dimension_0 + constant_0).as_ref(), is_equality: false };
        let constraint_1 = IntegerSetConstraint { expression: (dimension_0 + constant_1).as_ref(), is_equality: false };
        let set = context.integer_set(1, 0, &[&constraint_0, &constraint_1]);

        // We are just checking that [`IntegerSet::dump`] runs successfully without crashing.
        // Ideally, we would want a way to capture the standard error stream and verify that it printed the right thing.
        set.dump();
    }
}
