use std::fmt::{Debug, Display};

use ryft_xla_sys::bindings::{
    MlirAffineMap, mlirAffineMapCompressUnusedSymbols, mlirAffineMapConstantGet, mlirAffineMapDump,
    mlirAffineMapEmptyGet, mlirAffineMapEqual, mlirAffineMapGet, mlirAffineMapGetMajorSubMap,
    mlirAffineMapGetMinorSubMap, mlirAffineMapGetNumDims, mlirAffineMapGetNumInputs, mlirAffineMapGetNumResults,
    mlirAffineMapGetNumSymbols, mlirAffineMapGetResult, mlirAffineMapGetSingleConstantResult, mlirAffineMapGetSubMap,
    mlirAffineMapIsEmpty, mlirAffineMapIsIdentity, mlirAffineMapIsMinorIdentity, mlirAffineMapIsPermutation,
    mlirAffineMapIsProjectedPermutation, mlirAffineMapIsSingleConstant, mlirAffineMapMinorIdentityGet,
    mlirAffineMapMultiDimIdentityGet, mlirAffineMapPermutationGet, mlirAffineMapPrint, mlirAffineMapReplace,
    mlirAffineMapZeroResultGet,
};

use crate::{Context, support::write_to_formatter_callback};

use super::affine_expressions::{AffineExpression, AffineExpressionRef};

/// Multidimensional affine map. Affine maps are mathematical functions which map lists of dimensions, identifiers,
/// and symbols, to multidimensional affine expressions (e.g., `(d0, d1) -> (d0/128, d0 mod 128, d1)` where the names
/// being used here do not matter; it is the mathematical function that is unique to this affine map and that defines
/// it). They are immutable, uniqued, and always owned by a [`Context`].
#[derive(Copy, Clone)]
pub struct AffineMap<'c, 't> {
    /// Handle that represents this [`AffineMap`] in the MLIR C API.
    handle: MlirAffineMap,

    /// [`Context`] that owns this [`AffineMap`].
    context: &'c Context<'t>,
}

impl<'c, 't> AffineMap<'c, 't> {
    /// Constructs a new [`AffineMap`] from the provided [`MlirAffineMap`] handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirAffineMap, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context }) }
    }

    /// Returns the [`MlirAffineMap`] that corresponds to this [`AffineMap`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirAffineMap {
        self.handle
    }

    /// Returns a reference to the [`Context`] that owns this [`AffineMap`].
    pub fn context(&self) -> &'c Context<'t> {
        self.context
    }

    /// Returns the number of dimensions of this [`AffineMap`].
    pub fn dimension_count(&self) -> usize {
        unsafe { mlirAffineMapGetNumDims(self.handle).cast_unsigned() }
    }

    /// Returns the number of symbols of this [`AffineMap`].
    pub fn symbol_count(&self) -> usize {
        unsafe { mlirAffineMapGetNumSymbols(self.handle).cast_unsigned() }
    }

    /// Returns the number of inputs (i.e., number of dimensions and symbols combined) of this [`AffineMap`].
    pub fn input_count(&self) -> usize {
        unsafe { mlirAffineMapGetNumInputs(self.handle).cast_unsigned() }
    }

    /// Returns the number of results of this [`AffineMap`].
    pub fn result_count(&self) -> usize {
        unsafe { mlirAffineMapGetNumResults(self.handle).cast_unsigned() }
    }

    /// Returns the result [`AffineExpression`]s of this [`AffineMap`].
    pub fn results(&self) -> impl Iterator<Item = AffineExpressionRef<'c, 't>> {
        (0..self.result_count()).map(|index| self.result(index))
    }

    /// Returns the `index`-th result [`AffineExpression`] of this [`AffineMap`].
    ///
    /// Note that this function will panic if the provided index is out of bounds.
    pub fn result(&self, index: usize) -> AffineExpressionRef<'c, 't> {
        if index >= self.result_count() {
            panic!("result index is out of bounds");
        }
        unsafe {
            AffineExpressionRef::from_c_api(mlirAffineMapGetResult(self.handle, index.cast_signed()), self.context)
                .unwrap()
        }
    }

    /// Returns `true` if this map is an empty map. Refer to [`Context::empty_affine_map`] for more information.
    pub fn is_empty(&self) -> bool {
        unsafe { mlirAffineMapIsEmpty(self.handle) }
    }

    /// Returns `true` if this map is a single result constant map. Refer to [`Context::constant_affine_map`]
    /// for more information.
    pub fn is_constant(&self) -> bool {
        unsafe { mlirAffineMapIsSingleConstant(self.handle) }
    }

    /// Returns `true` if this map is an identity map. This function also asserts that the number of dimensions in this
    /// map is greater than or equal to its number of results.
    pub fn is_identity(&self) -> bool {
        unsafe { mlirAffineMapIsIdentity(self.handle) }
    }

    /// Returns `true` if this map is a minor identity map. Refer to [`Context::minor_identity_affine_map`]
    /// for more information.
    pub fn is_minor_identity(&self) -> bool {
        unsafe { mlirAffineMapIsMinorIdentity(self.handle) }
    }

    /// Returns `true` if this map represents a "symbol-less" permutation map.
    /// Refer to [`Context::permutation_affine_map`] for more information.
    pub fn is_permutation(&self) -> bool {
        unsafe { mlirAffineMapIsPermutation(self.handle) }
    }

    /// Returns `true` if this map represents a subset of a "symbol-less" permutation map.
    /// Refer to [`Context::permutation_affine_map`] for more information.
    pub fn is_projected_permutation(&self) -> bool {
        unsafe { mlirAffineMapIsProjectedPermutation(self.handle) }
    }

    /// Returns the constant result of this map, if it is a single result constant map, and [`None`] otherwise.
    /// Refer to [`Context::constant_affine_map`] for more information.
    pub fn constant(&self) -> Option<i64> {
        if !self.is_constant() { None } else { Some(unsafe { mlirAffineMapGetSingleConstantResult(self.handle) }) }
    }

    /// Returns the [`AffineMap`] that consists only of the results at the specified indices of this [`AffineMap`].
    pub fn sub_map(&self, size: usize, results: &[usize]) -> Self {
        unsafe {
            Self::from_c_api(
                mlirAffineMapGetSubMap(self.handle, size.cast_signed(), results.as_ptr() as *mut _),
                self.context,
            )
            .unwrap()
        }
    }

    /// Returns the [`AffineMap`] that consists only of the most major `result_count` results of this [`AffineMap`].
    /// Returns [`None`] if `result_count` is set to zero and returns `self` if it is set to a number that is greater
    /// than or equal to the number of results of this [`AffineMap`].
    pub fn major_sub_map(&self, result_count: usize) -> Option<Self> {
        unsafe { Self::from_c_api(mlirAffineMapGetMajorSubMap(self.handle, result_count.cast_signed()), self.context) }
    }

    /// Returns the [`AffineMap`] that consists only of the most minor `result_count` results of this [`AffineMap`].
    /// Returns [`None`] if `result_count` is set to zero and returns `self` if it is set to a number that is greater
    /// than or equal to the number of results of this [`AffineMap`].
    pub fn minor_sub_map(&self, result_count: usize) -> Option<Self> {
        unsafe { Self::from_c_api(mlirAffineMapGetMinorSubMap(self.handle, result_count.cast_signed()), self.context) }
    }

    /// Replaces `expression` with `replacement` in each of the results of this [`AffineMap`], returning the resulting
    /// [`AffineMap`], with the provided number of dimensions and symbols.
    pub fn replace<E: AffineExpression<'c, 't>, R: AffineExpression<'c, 't>>(
        &self,
        expression: E,
        replacement: R,
        result_dimensions_count: usize,
        result_symbols_count: usize,
    ) -> Self {
        unsafe {
            Self::from_c_api(
                mlirAffineMapReplace(
                    self.handle,
                    expression.to_c_api(),
                    replacement.to_c_api(),
                    result_dimensions_count.cast_signed(),
                    result_symbols_count.cast_signed(),
                ),
                self.context,
            )
            .unwrap()
        }
    }

    /// Dumps this [`AffineMap`] to the standard error stream.
    pub fn dump(&self) {
        unsafe { mlirAffineMapDump(self.handle) }
    }

    /// Compresses (i.e., simplifies) the provided [`AffineMap`]s by dropping symbols that do not appear in any of
    /// them. This function also asserts that all of the provided maps are normalized to the same number of dimensions
    /// and symbols.
    pub fn compress_unused_symbols<'m: 'c, 's: 'm>(
        maps: &'m [&'s Self],
    ) -> impl Iterator<Item = Self> + use<'c, 't, 's> {
        extern "C" fn populate_result(results: *mut std::ffi::c_void, index: isize, map: MlirAffineMap) {
            unsafe {
                let offset = index.cast_unsigned() * size_of::<MlirAffineMap>();
                let result = results.offset(offset.cast_signed()) as *mut MlirAffineMap;
                *result = map;
            }
        }

        unsafe {
            let map_handles = maps.iter().map(|map| map.to_c_api()).collect::<Vec<_>>();
            let mut buffer: Vec<MlirAffineMap> = Vec::with_capacity(maps.len());
            mlirAffineMapCompressUnusedSymbols(
                map_handles.as_ptr() as _,
                map_handles.len().cast_signed(),
                buffer.as_mut_ptr() as *mut _,
                Some(populate_result),
            );
            buffer.set_len(maps.len());
            buffer
                .into_iter()
                .zip(maps.iter())
                .map(|(compressed_map, original_map)| Self::from_c_api(compressed_map, original_map.context).unwrap())
        }
    }
}

impl PartialEq for AffineMap<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirAffineMapEqual(self.handle, other.handle) }
    }
}

impl Eq for AffineMap<'_, '_> {}

impl Display for AffineMap<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirAffineMapPrint(
                self.handle,
                Some(write_to_formatter_callback),
                &mut data as *mut _ as *mut std::ffi::c_void,
            );
        }
        data.1
    }
}

impl Debug for AffineMap<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "AffineMap[{self}]")
    }
}

impl<'t> Context<'t> {
    /// Creates an [`AffineMap`] with the results defined by the provided [`AffineExpression`]s. The resulting map also
    /// has the specified number of dimensions and symbols, regardless of them being used in the provided expressions.
    /// The resulting map is owned by this [`Context`].
    pub fn affine_map<'c, A: AffineExpression<'c, 't>>(
        &'c self,
        dimension_count: usize,
        symbol_count: usize,
        affine_expressions: &[A],
    ) -> AffineMap<'c, 't> {
        unsafe {
            let affine_expressions = affine_expressions.iter().map(|e| e.to_c_api()).collect::<Vec<_>>();
            AffineMap::from_c_api(
                mlirAffineMapGet(
                    *self.handle.borrow_mut(),
                    dimension_count.cast_signed(),
                    symbol_count.cast_signed(),
                    affine_expressions.len().cast_signed(),
                    affine_expressions.as_ptr() as *mut _,
                ),
                self,
            )
            .unwrap()
        }
    }

    /// Creates an empty [`AffineMap`] (i.e., a zero result affine map with no dimensions or symbols; `() -> ()`)
    /// owned by this [`Context`].
    pub fn empty_affine_map<'c>(&'c self) -> AffineMap<'c, 't> {
        unsafe { AffineMap::from_c_api(mlirAffineMapEmptyGet(*self.handle.borrow_mut()), self).unwrap() }
    }

    /// Creates a zero result [`AffineMap`] with the provided number of dimensions and symbols (i.e., `(...) -> ()`),
    /// owned by this [`Context`].
    pub fn zero_result_affine_map<'c>(&'c self, dimension_count: usize, symbol_count: usize) -> AffineMap<'c, 't> {
        unsafe {
            AffineMap::from_c_api(
                mlirAffineMapZeroResultGet(
                    *self.handle.borrow_mut(),
                    dimension_count.cast_signed(),
                    symbol_count.cast_signed(),
                ),
                self,
            )
            .unwrap()
        }
    }

    /// Creates an [`AffineMap`] with a single constant result. The resulting map is owned by this [`Context`].
    pub fn constant_affine_map<'c>(&'c self, value: i64) -> AffineMap<'c, 't> {
        unsafe { AffineMap::from_c_api(mlirAffineMapConstantGet(*self.handle.borrow_mut(), value), self).unwrap() }
    }

    /// Creates a multidimensional identity [`AffineMap`] with the specified number of dimensions.
    /// The resulting map is owned by this [`Context`].
    pub fn identity_affine_map<'c>(&'c self, dimension_count: usize) -> AffineMap<'c, 't> {
        unsafe {
            AffineMap::from_c_api(
                mlirAffineMapMultiDimIdentityGet(*self.handle.borrow_mut(), dimension_count.cast_signed()),
                self,
            )
            .unwrap()
        }
    }

    /// Creates a multidimensional identity [`AffineMap`] on the most minor dimensions for the specified number of
    /// dimensions and results (where the number of dimensions must be greater than or equal to the number of results).
    /// The resulting map is owned by this [`Context`].
    pub fn minor_identity_affine_map<'c>(&'c self, dimension_count: usize, result_count: usize) -> AffineMap<'c, 't> {
        unsafe {
            AffineMap::from_c_api(
                mlirAffineMapMinorIdentityGet(
                    *self.handle.borrow_mut(),
                    dimension_count.cast_signed(),
                    result_count.cast_signed(),
                ),
                self,
            )
            .unwrap()
        }
    }

    /// Creates an [`AffineMap`] based on the provided permutation which contains a permutation of indexes starting
    /// from 0 and ending at `size - 1` (e.g., `[1, 2, 0]` is a valid permutation but `[2, 0]` and `[1, 1, 2]` are
    /// not valid permutations). The resulting map is owned by this [`Context`].
    pub fn permutation_affine_map<'c>(&'c self, size: usize, permutation: &[usize]) -> AffineMap<'c, 't> {
        unsafe {
            let permutation = permutation.iter().map(|value| *value as std::ffi::c_uint).collect::<Vec<_>>();
            AffineMap::from_c_api(
                mlirAffineMapPermutationGet(
                    *self.handle.borrow_mut(),
                    size.cast_signed(),
                    permutation.as_ptr() as *mut _,
                ),
                self,
            )
            .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Context;

    use super::*;

    #[test]
    fn test_affine_map() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let constant_0 = context.constant_affine_expression(2);
        let expression_0 = (dimension_0 + dimension_1).as_ref();
        let expression_1 = (dimension_0 * constant_0).as_ref();

        let map = context.affine_map(0, 0, &[constant_0]);
        assert_eq!(&context, map.context());
        assert_eq!(map.dimension_count(), 0);
        assert_eq!(map.symbol_count(), 0);
        assert_eq!(map.input_count(), 0);
        assert_eq!(map.result_count(), 1);
        assert!(map.is_constant());
        assert_eq!(map.constant(), Some(2));
        assert_eq!(map.to_string(), "() -> (2)");

        let map = context.affine_map(2, 1, &[expression_0, expression_1]);
        assert_eq!(&context, map.context());
        assert_eq!(map.dimension_count(), 2);
        assert_eq!(map.symbol_count(), 1);
        assert_eq!(map.input_count(), 3);
        assert_eq!(map.result_count(), 2);
        assert!(!map.is_constant());
        assert_eq!(map.constant(), None);
        assert_eq!(map.to_string(), "(d0, d1)[s0] -> (d0 + d1, d0 * 2)");
    }

    #[test]
    fn test_affine_map_results() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let result_0 = (dimension_0 + dimension_1).as_ref();
        let result_1 = dimension_0.as_ref();
        let map = context.affine_map(2, 0, &[result_0, result_1]);
        assert_eq!(map.result(0), result_0);
        assert_eq!(map.result(1), result_1);
        assert_eq!(map.results().collect::<Vec<_>>(), vec![result_0, result_1]);
    }

    #[test]
    fn test_empty_affine_map() {
        let context = Context::new();
        let map = context.empty_affine_map();
        assert_eq!(&context, map.context());
        assert_eq!(map.dimension_count(), 0);
        assert_eq!(map.symbol_count(), 0);
        assert_eq!(map.input_count(), 0);
        assert_eq!(map.result_count(), 0);
        assert!(map.is_empty());
        assert_eq!(map.to_string(), "() -> ()");
    }

    #[test]
    fn test_zero_result_affine_map() {
        let context = Context::new();
        let map = context.zero_result_affine_map(3, 2);
        assert_eq!(map.dimension_count(), 3);
        assert_eq!(map.symbol_count(), 2);
        assert_eq!(map.result_count(), 0);
        assert_eq!(map.to_string(), "(d0, d1, d2)[s0, s1] -> ()");
    }

    #[test]
    fn test_constant_affine_map() {
        let context = Context::new();
        let map_0 = context.constant_affine_map(42);
        let map_1 = context.constant_affine_map(-5);
        assert_eq!(map_0.dimension_count(), 0);
        assert_eq!(map_0.symbol_count(), 0);
        assert_eq!(map_0.result_count(), 1);
        assert_eq!(map_0.to_string(), "() -> (42)");
        assert_eq!(map_1.to_string(), "() -> (-5)");
    }

    #[test]
    fn test_identity_affine_map() {
        let context = Context::new();
        let map = context.identity_affine_map(3);
        assert_eq!(map.dimension_count(), 3);
        assert_eq!(map.result_count(), 3);
        assert!(map.is_identity());
        assert_eq!(map.to_string(), "(d0, d1, d2) -> (d0, d1, d2)");
    }

    #[test]
    fn test_minor_identity_affine_map() {
        let context = Context::new();
        let map = context.minor_identity_affine_map(5, 3);
        assert_eq!(map.dimension_count(), 5);
        assert_eq!(map.result_count(), 3);
        assert!(map.is_minor_identity());
        assert_eq!(map.to_string(), "(d0, d1, d2, d3, d4) -> (d2, d3, d4)");
    }

    #[test]
    fn test_permutation_affine_map() {
        let context = Context::new();
        let permutation = vec![2, 0, 1];
        let map = context.permutation_affine_map(3, &permutation);
        assert_eq!(map.dimension_count(), 3);
        assert_eq!(map.result_count(), 3);
        assert!(map.is_permutation());
        assert_eq!(map.to_string(), "(d0, d1, d2) -> (d2, d0, d1)");
    }

    #[test]
    fn test_affine_map_is_projected_permutation() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(2);
        let permutation_map = context.permutation_affine_map(3, &[2, 0, 1]);
        let projected_map = context.affine_map(3, 0, &[dimension_0, dimension_1]);
        assert!(permutation_map.is_projected_permutation());
        assert!(projected_map.is_projected_permutation());
    }

    #[test]
    fn test_affine_map_equality() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let expression = dimension_0 + dimension_1;

        // Same maps from the same context must be equal because they are "uniqued".
        let map_0 = context.affine_map(2, 0, &[expression]);
        let map_1 = context.affine_map(2, 0, &[expression]);
        assert_eq!(map_0, map_1);

        // Different results must not be equal.
        let map_2 = context.affine_map(2, 0, &[dimension_0]);
        assert_ne!(map_0, map_2);

        // Same maps from different contexts must not be equal.
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let map_3 = context.affine_map(2, 0, &[dimension_0 + dimension_1]);
        assert_ne!(map_0, map_3);
    }

    #[test]
    fn test_affine_map_sub_map() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0).as_ref();
        let dimension_1 = context.dimension_affine_expression(1).as_ref();
        let constant_0 = context.constant_affine_expression(2).as_ref();
        let expression = (dimension_0 * constant_0).as_ref();
        let map = context.affine_map(2, 0, &[dimension_0, dimension_1, expression]);
        let sub_map = map.sub_map(2, &[0, 2]);
        assert_eq!(sub_map.result_count(), 2);
        assert_eq!(sub_map.to_string(), "(d0, d1) -> (d0, d0 * 2)");
    }

    #[test]
    fn test_affine_map_major_sub_map() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let dimension_2 = context.dimension_affine_expression(2);
        let map = context.affine_map(3, 0, &[dimension_0, dimension_1, dimension_2]);
        let major_sub_map = map.major_sub_map(2).unwrap();
        assert_eq!(major_sub_map.result_count(), 2);
        assert_eq!(major_sub_map.to_string(), "(d0, d1, d2) -> (d0, d1)");
        assert!(map.major_sub_map(0).is_none());
        assert_eq!(map.major_sub_map(5).unwrap(), map);
    }

    #[test]
    fn test_affine_map_minor_sub_map() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let dimension_2 = context.dimension_affine_expression(2);
        let map = context.affine_map(3, 0, &[dimension_0, dimension_1, dimension_2]);
        let minor_sub_map = map.minor_sub_map(2).unwrap();
        assert_eq!(minor_sub_map.result_count(), 2);
        assert_eq!(minor_sub_map.to_string(), "(d0, d1, d2) -> (d1, d2)");
        assert!(map.minor_sub_map(0).is_none());
        assert_eq!(map.minor_sub_map(5).unwrap(), map);
    }

    #[test]
    fn test_affine_map_replace() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let constant_0 = context.constant_affine_expression(2);
        let map = context.affine_map(2, 0, &[dimension_0 + dimension_1]);
        let replaced = map.replace(dimension_0, dimension_1 * constant_0, 2, 0);
        assert_eq!(replaced.to_string(), "(d0, d1) -> (d1 * 3)");
    }

    #[test]
    fn test_affine_map_display_and_debug() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let map = context.affine_map(2, 0, &[dimension_0 + dimension_1]);
        assert_eq!(format!("{}", map), "(d0, d1) -> (d0 + d1)");
        assert_eq!(format!("{:?}", map), "AffineMap[(d0, d1) -> (d0 + d1)]");
    }
    #[test]
    fn test_affine_map_dump() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let constant_0 = context.constant_affine_expression(2);
        let expression_0 = (dimension_0 + dimension_1).as_ref();
        let expression_1 = (dimension_0 * constant_0).as_ref();
        let map = context.affine_map(2, 1, &[expression_0, expression_1]);

        // We are just checking that [`AffineMap::dump`] runs successfully without crashing.
        // Ideally, we would want a way to capture the standard error stream and verify that it printed the right thing.
        map.dump();
    }

    #[test]
    fn test_affine_map_compress_unused_symbols() {
        let context = Context::new();
        let dimension_0 = context.dimension_affine_expression(0);
        let dimension_1 = context.dimension_affine_expression(1);
        let constant_0 = context.constant_affine_expression(2);
        let symbol_0 = context.symbol_affine_expression(0);
        let expression_0 = (dimension_0 + dimension_1).as_ref();
        let expression_1 = (dimension_0 * constant_0).as_ref();
        let map_0 = context.affine_map(4, 2, &[expression_0, expression_1]);
        let map_1 = context.affine_map(2, 4, &[symbol_0 * constant_0]);
        let maps = [&map_0, &map_1];
        assert_eq!(maps[0].to_string(), "(d0, d1, d2, d3)[s0, s1] -> (d0 + d1, d0 * 2)");
        assert_eq!(maps[1].to_string(), "(d0, d1)[s0, s1, s2, s3] -> (s0 * 2)");
        let compressed_maps = AffineMap::compress_unused_symbols(&maps).collect::<Vec<_>>();
        assert_eq!(maps[0].to_string(), "(d0, d1, d2, d3)[s0, s1] -> (d0 + d1, d0 * 2)");
        assert_eq!(maps[1].to_string(), "(d0, d1)[s0, s1, s2, s3] -> (s0 * 2)");
        assert_eq!(compressed_maps.len(), 2);
        assert_eq!(compressed_maps[0].to_string(), "(d0, d1, d2, d3)[s0] -> (d0 + d1, d0 * 2)");
        assert_eq!(compressed_maps[1].to_string(), "(d0, d1, d2, d3)[s0] -> (s0 * 2)");
    }
}
