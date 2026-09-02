use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Range;

use ryft_xla_sys::bindings::{
    MlirOperation, MlirWalkOrder, MlirWalkOrder_MlirWalkPostOrder, MlirWalkOrder_MlirWalkPreOrder, MlirWalkResult,
    MlirWalkResult_MlirWalkResultAdvance, MlirWalkResult_MlirWalkResultInterrupt, MlirWalkResult_MlirWalkResultSkip,
    mlirOperationClone, mlirOperationCreateParse, mlirOperationDestroy, mlirOperationDump, mlirOperationEqual,
    mlirOperationGetAttribute, mlirOperationGetAttributeByName, mlirOperationGetBlock,
    mlirOperationGetDiscardableAttribute, mlirOperationGetDiscardableAttributeByName,
    mlirOperationGetInherentAttributeByName, mlirOperationGetLocation, mlirOperationGetName,
    mlirOperationGetNumAttributes, mlirOperationGetNumDiscardableAttributes, mlirOperationGetNumOperands,
    mlirOperationGetNumRegions, mlirOperationGetNumResults, mlirOperationGetNumSuccessors, mlirOperationGetOpOperand,
    mlirOperationGetOperand, mlirOperationGetParentOperation, mlirOperationGetRegion, mlirOperationGetResult,
    mlirOperationGetSuccessor, mlirOperationGetTypeID, mlirOperationHasInherentAttributeByName, mlirOperationHashValue,
    mlirOperationIsBeforeInBlock, mlirOperationMoveAfter, mlirOperationMoveBefore, mlirOperationPrint,
    mlirOperationPrintWithFlags, mlirOperationPrintWithState, mlirOperationRemoveAttributeByName,
    mlirOperationRemoveDiscardableAttributeByName, mlirOperationReplaceUsesOfWith, mlirOperationSetAttributeByName,
    mlirOperationSetDiscardableAttributeByName, mlirOperationSetInherentAttributeByName, mlirOperationSetLocation,
    mlirOperationSetOperand, mlirOperationSetOperands, mlirOperationSetSuccessor, mlirOperationVerify,
    mlirOperationWalk, mlirOperationWriteBytecode, mlirOperationWriteBytecodeWithConfig,
};

use crate::support::{write_to_formatter_callback, write_to_string_callback};
use crate::{
    AffineMapAttributeRef, ArrayAttributeRef, Attribute, AttributeRef, Block, BlockRef, BooleanAttributeRef, Context,
    DenseArrayAttributeRef, DenseBooleanArrayAttributeRef, DenseElementsAttributeRef, DenseFloat32ArrayAttributeRef,
    DenseFloat64ArrayAttributeRef, DenseFloatElementsAttributeRef, DenseInteger8ArrayAttributeRef,
    DenseInteger16ArrayAttributeRef, DenseInteger32ArrayAttributeRef, DenseInteger64ArrayAttributeRef,
    DenseIntegerElementsAttributeRef, DenseResourceElementsAttributeRef, DictionaryAttributeRef, DistinctAttributeRef,
    ElementsAttributeRef, Error, FlatSymbolRefAttributeRef, FloatAttributeRef, Identifier, IntegerAttributeRef,
    IntegerSetAttributeRef, Location, LocationAttributeRef, LocationRef, LogicalResult, NamedAttributeRef,
    OpaqueAttributeRef, OperandRef, OperationResultRef, RegionRef, SparseElementsAttributeRef,
    StridedLayoutAttributeRef, StringAttributeRef, StringRef, SymbolRefAttributeRef, SymbolVisibilityAttributeRef,
    TypeAttributeRef, TypeId, TypeRef, UnitAttributeRef, Value, ValueRef, write_to_bytes_callback,
};

use super::printing::{AsmState, BytecodeWriterConfiguration, OperationPrintingFlags};

/// Helper macro for defining typed attribute accessor functions for [`Operation`]s.
macro_rules! mlir_operation_builtin_attribute {
    ($method_name:ident, $attribute_type:ty) => {
        /// Returns an [`Attribute`] of this [`Operation`] with the provided name. If no such attribute
        /// can be found or if it is not of the appropriate type, then this function returns an [`Error`].
        fn $method_name<N: AsRef<str>>(&self, name: N) -> Result<$attribute_type, Error> {
            let name = name.as_ref();
            self.attribute(name)?.and_then(|attribute| attribute.cast::<$attribute_type>()).ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    name,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })
        }
    };
}

/// [`Operation`]s are one of the main building blocks of MLIR programs. MLIR is fundamentally based on a graph-like
/// data structure of nodes, called [`Operation`]s, and edges, called [`Value`]s. Each [`Value`] is either a
/// [`BlockArgumentRef`](crate::BlockArgumentRef) or an [`OperationResultRef`], and has a [`Type`](crate::Type)
/// defined by the type system. [`Operation`]s are contained in [`Block`]s and [`Block`]s are contained in
/// [`Region`](crate::Region)s. [`Operation`]s are also ordered within their containing [`Block`] and [`Block`]s
/// are ordered in their containing [`Region`](crate::Region)s, although this order may or may not be semantically
/// meaningful in a given kind of region). [`Operation`]s may also contain [`Region`](crate::Region)s, enabling
/// hierarchical structures to be represented.
///
/// Note that there are multiple separate lifetime parameters: one for the lifetime of the underlying [`Operation`],
/// `'o`, one for the [`Context`] which is associated with it, `'c`, and one for the lifetime of the thread pool used
/// by that [`Context`], `'t`. That is because, [`Operation`]s can be either owned (i.e., [`DetachedOperation`]s) or
/// borrowed references to underlying MLIR operations owned by [`Block`]s (i.e., [`OperationRef`]s), which themselves
/// may be owned by [`Region`](crate::Region)s, etc.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/LangRef/#high-level-structure)
/// for more information.
pub trait Operation<'o, 'c: 'o, 't: 'c>: Sized {
    /// Constructs a new [`Operation`] of this type from the provided handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn from_c_api(handle: MlirOperation, context: &'c Context<'t>) -> Result<Self, Error>;

    /// Returns the [`MlirOperation`] that corresponds to this [`Operation`] and which can be passed to functions
    /// in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn to_c_api(&self) -> MlirOperation;

    /// Returns a reference to the [`Context`] that is associated with this [`Operation`].
    fn context(&self) -> &'c Context<'t>;

    /// Returns an [`OperationRef`] that references this [`Operation`].
    fn as_ref(&self) -> OperationRef<'o, 'c, 't> {
        OperationRef { handle: unsafe { self.to_c_api() }, context: self.context(), owner: PhantomData }
    }

    /// Gets the [`TypeId`] of this [`Operation`]. Note that this function may return the same [`TypeId`] for different
    /// instances of the same operation with potentially different attributes. That is because a [`TypeId`] is a unique
    /// identifier of the corresponding MLIR C++ type for the operation and not for a specific instance of this
    /// operation type. Also, note that if the operation does not have a registered description, then this function
    /// will return [`None`].
    fn type_id(&self) -> Result<Option<TypeId<'c>>, Error> {
        unsafe {
            let handle = mlirOperationGetTypeID(self.to_c_api());
            if handle.ptr.is_null() { Ok(None) } else { TypeId::from_c_api(handle).map(Some) }
        }
    }

    /// Returns the [`Location`] of this [`Operation`].
    fn location(&self) -> Result<LocationRef<'c, 't>, Error> {
        unsafe { LocationRef::from_c_api(mlirOperationGetLocation(self.to_c_api()), self.context()) }
    }

    /// Sets the [`Location`] of this [`Operation`].
    fn set_location<L: Location<'c, 't>>(&mut self, location: L) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe { mlirOperationSetLocation(self.to_c_api(), location.to_c_api()) }
    }

    /// Returns the name of this [`Operation`].
    fn name(&self) -> Identifier<'c, 't> {
        unsafe { Identifier::from_c_api(mlirOperationGetName(self.to_c_api())) }
    }

    /// Returns the number of inherent [`Attribute`]s of this [`Operation`]. Refer to the documentation of
    /// [`Operation::attributes`] for information on the distinction between inherent and
    /// discardable attributes.
    fn inherent_attribute_count(&self) -> usize {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        self.attribute_count() - self.discardable_attribute_count()
    }

    /// Returns `true` if this [`Operation`] has an inherent [`Attribute`] with the provided name (even if the
    /// attribute is optional, meaning that [`Operation::inherent_attribute`] could still return [`None`] in
    /// that case).
    ///
    /// Refer to the documentation of [`Operation::attributes`] for information on the distinction between
    /// inherent and discardable attributes.
    fn has_inherent_attribute<N: AsRef<str>>(&self, name: N) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirOperationHasInherentAttributeByName(self.to_c_api(), StringRef::from(name.as_ref()).to_c_api()) }
    }

    /// Returns an inherent [`Attribute`] of this [`Operation`] with the provided name. If no such attribute can be
    /// found, then this function returns `Ok(None)`.
    ///
    /// Refer to the documentation of [`Operation::attributes`] for information on the distinction between
    /// inherent and discardable attributes.
    fn inherent_attribute<N: AsRef<str>>(&self, name: N) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        let handle = unsafe {
            mlirOperationGetInherentAttributeByName(self.to_c_api(), StringRef::from(name.as_ref()).to_c_api())
        };
        if handle.ptr.is_null() {
            Ok(None)
        } else {
            unsafe { AttributeRef::from_c_api(handle, self.context()).map(Some) }
        }
    }

    /// Sets an inherent attribute of this [`Operation`] with the provided name to the provided value. This function
    /// will do nothing if this operation does not have an inherent attribute with the specified name.
    ///
    /// Refer to the documentation of [`Operation::attributes`] for information on the distinction between
    /// inherent and discardable attributes.
    fn set_inherent_attribute<N: AsRef<str>, A: Attribute<'c, 't>>(&mut self, name: N, value: A) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            mlirOperationSetInherentAttributeByName(
                self.to_c_api(),
                StringRef::from(name.as_ref()).to_c_api(),
                value.to_c_api(),
            )
        }
    }

    /// Returns the number of discardable [`Attribute`]s of this [`Operation`]. Refer to the documentation of
    /// [`Operation::attributes`] for information on the distinction between inherent and
    /// discardable attributes.
    fn discardable_attribute_count(&self) -> usize {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirOperationGetNumDiscardableAttributes(self.to_c_api()).cast_unsigned() }
    }

    /// Returns an [`Iterator`] over the discardable [`Attribute`]s of this [`Operation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`]
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    ///
    /// Refer to the documentation of [`Operation::attributes`] for information on the distinction between
    /// inherent and discardable attributes.
    fn discardable_attributes<'r>(&'r self) -> impl Iterator<Item = NamedAttributeRef<'c, 't>> {
        (0..self.discardable_attribute_count()).map(|index| unsafe {
            NamedAttributeRef::from_c_api(
                mlirOperationGetDiscardableAttribute(self.to_c_api(), index.cast_signed()),
                self.context(),
            )
        })
    }

    /// Returns `true` if this [`Operation`] has a discardable [`Attribute`] with the provided name.
    ///
    /// Refer to the documentation of [`Operation::attributes`] for information on the distinction between
    /// inherent and discardable attributes.
    fn has_discardable_attribute<N: AsRef<str>>(&self, name: N) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        let handle = unsafe {
            mlirOperationGetDiscardableAttributeByName(self.to_c_api(), StringRef::from(name.as_ref()).to_c_api())
        };
        !handle.ptr.is_null()
    }

    /// Returns a discardable [`Attribute`] of this [`Operation`] with the provided name. If no such attribute can be
    /// found, then this function returns `Ok(None)`.
    ///
    /// Refer to the documentation of [`Operation::attributes`] for information on the distinction between
    /// inherent and discardable attributes.
    fn discardable_attribute<N: AsRef<str>>(&self, name: N) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        let handle = unsafe {
            mlirOperationGetDiscardableAttributeByName(self.to_c_api(), StringRef::from(name.as_ref()).to_c_api())
        };
        if handle.ptr.is_null() {
            Ok(None)
        } else {
            unsafe { AttributeRef::from_c_api(handle, self.context()).map(Some) }
        }
    }

    /// Sets a discardable attribute of this [`Operation`] with the provided name to the provided value. Note that if
    /// the provided value is a `null` [`Attribute`], then this function will remove that attribute from the operation.
    ///
    /// Refer to the documentation of [`Operation::attributes`] for information on the distinction between
    /// inherent and discardable attributes.
    fn set_discardable_attribute<N: AsRef<str>, A: Attribute<'c, 't>>(&mut self, name: N, value: A) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            let value_handle = value.to_c_api();
            if !value_handle.ptr.is_null() {
                mlirOperationSetDiscardableAttributeByName(
                    self.to_c_api(),
                    StringRef::from(name.as_ref()).to_c_api(),
                    value_handle,
                )
            } else {
                mlirOperationRemoveDiscardableAttributeByName(
                    self.to_c_api(),
                    StringRef::from(name.as_ref()).to_c_api(),
                );
            }
        }
    }

    /// Removes the discardable attribute of this [`Operation`] with the provided name, returning `true` if the
    /// attribute was removed successfully and `false` otherwise (e.g., if no such attribute could be found).
    ///
    /// Refer to the documentation of [`Operation::attributes`] for information on the distinction between
    /// inherent and discardable attributes.
    fn remove_discardable_attribute<N: AsRef<str>>(&mut self, name: N) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            mlirOperationRemoveDiscardableAttributeByName(self.to_c_api(), StringRef::from(name.as_ref()).to_c_api())
        }
    }

    /// Returns the number of [`Attribute`]s of this [`Operation`], including both inherent and discardable attributes.
    /// Refer to the documentation of [`Operation::attributes`] for information on this distinction.
    fn attribute_count(&self) -> usize {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirOperationGetNumAttributes(self.to_c_api()).cast_unsigned() }
    }

    /// Returns an [`Iterator`] over the [`Attribute`]s of this [`Operation`] (along with their names). Note that
    /// attributes may be dynamically added and removed over the lifetime of an operation.
    ///
    /// MLIR makes a distinction between **inherent** and **discardable** attributes of operations which relates to how
    /// essential they are to the operation's semantics and identity. This function returns all attributes of this
    /// operation, ignoring this distinction. Therefore, it is recommended to instead rely on accessing inherent
    /// attributes by name using [`Operation::inherent_attribute`] and discardable attributes using
    /// [`Operation::discardable_attributes`] or [`Operation::discardable_attribute`].
    ///
    /// # Inherent Attributes
    ///
    /// Inherent attributes are fundamental to an operation's meaning and behavior. They:
    ///
    ///   - are part of the operation's core semantics and cannot be removed without changing what it does,
    ///   - are typically defined as part of the operation's specification,
    ///   - must be preserved across transformations to maintain correctness, and
    ///   - are used by the operation's verifier, folder, canonicalizer, and other core functionality.
    ///
    /// Examples include:
    ///
    ///   - the `value` attribute of `arith.constant`,
    ///   - the `callee` attribute of `func.call`,
    ///   - loop bounds in structured control flow operations, and
    ///   - predicate types in comparison operations.
    ///
    /// # Discardable Attributes
    ///
    /// Discardable attributes are auxiliary metadata that can be safely removed without affecting the operation's
    /// core semantics. They:
    ///
    ///   - provide additional information that may be useful for optimization, debugging, or analysis,
    ///   - can be dropped by transformations without breaking correctness,
    ///   - are often added by passes for bookkeeping or to communicate information between passes, and
    ///   - have names that typically start with a dialect prefix to avoid conflicts.
    ///
    /// Examples include:
    ///
    ///   - debug information (e.g., [`LocationAttributeRef`](crate::LocationAttributeRef)s),
    ///   - optimization hints that do not change semantics,
    ///   - analysis results stored as attributes, and
    ///   - custom metadata added by specific passes.
    ///
    /// # Practical Implications
    ///
    /// This distinction is important because of the following reasons:
    ///
    ///   - **Transformation Safety:** Passes can freely remove discardable attributes but must preserve inherent ones.
    ///   - **Operation Equality:** Two operations with the same inherent attributes but different discardable
    ///     attributes may be considered semantically equivalent.
    ///   - **Serialization:** Some serialization formats might choose to omit discardable attributes to reduce size.
    ///   - **Verification:** Only inherent attributes are typically checked by the operation's verifier.
    ///
    /// Note that for unregistered operations that are not storing inherent attributes as properties, all attributes
    /// are considered discardable.
    ///
    /// Also note that the returned iterator does not hold a borrowed reference to the underlying [`Context`]
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    ///
    /// The MLIR infrastructure uses this distinction to enable safe and aggressive optimizations while maintaining
    /// semantic correctness.
    fn attributes<'r>(&'r self) -> impl Iterator<Item = NamedAttributeRef<'c, 't>> {
        (0..self.attribute_count()).map(|index| unsafe {
            NamedAttributeRef::from_c_api(
                mlirOperationGetAttribute(self.to_c_api(), index.cast_signed()),
                self.context(),
            )
        })
    }

    /// Returns `true` if this [`Operation`] has an [`Attribute`] with the provided name.
    ///
    /// It is recommended to instead use [`Operation::has_inherent_attribute`] or
    /// [`Operation::has_discardable_attribute`]. Refer to the documentation of [`Operation::attributes`] for
    /// information on the distinction between inherent and discardable attributes.
    fn has_attribute<N: AsRef<str>>(&self, name: N) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        let handle =
            unsafe { mlirOperationGetAttributeByName(self.to_c_api(), StringRef::from(name.as_ref()).to_c_api()) };
        !handle.ptr.is_null()
    }

    /// Returns an [`Attribute`] of this [`Operation`] with the provided name. If no such attribute can be found,
    /// then this function returns `Ok(None)`.
    ///
    /// It is recommended to instead use [`Operation::inherent_attribute`] or
    /// [`Operation::discardable_attribute`]. Refer to the documentation of [`Operation::attributes`] for information
    /// on the distinction between inherent and discardable attributes.
    fn attribute<N: AsRef<str>>(&self, name: N) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        let handle =
            unsafe { mlirOperationGetAttributeByName(self.to_c_api(), StringRef::from(name.as_ref()).to_c_api()) };
        if handle.ptr.is_null() {
            Ok(None)
        } else {
            unsafe { AttributeRef::from_c_api(handle, self.context()).map(Some) }
        }
    }

    mlir_operation_builtin_attribute!(affine_map_attribute, AffineMapAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(array_attribute, ArrayAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(boolean_attribute, BooleanAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(dense_array_attribute, DenseArrayAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(dense_boolean_array_attribute, DenseBooleanArrayAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(dense_integer_8_array_attribute, DenseInteger8ArrayAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(dense_integer_16_array_attribute, DenseInteger16ArrayAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(dense_integer_32_array_attribute, DenseInteger32ArrayAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(dense_integer_64_array_attribute, DenseInteger64ArrayAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(dense_float_32_array_attribute, DenseFloat32ArrayAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(dense_float_64_array_attribute, DenseFloat64ArrayAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(dictionary_attribute, DictionaryAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(distinct_attribute, DistinctAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(elements_attribute, ElementsAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(dense_elements_attribute, DenseElementsAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(dense_integer_elements_attribute, DenseIntegerElementsAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(dense_float_elements_attribute, DenseFloatElementsAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(dense_resource_elements_attribute, DenseResourceElementsAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(sparse_elements_attribute, SparseElementsAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(flat_symbol_ref_attribute, FlatSymbolRefAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(float_attribute, FloatAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(integer_attribute, IntegerAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(integer_set_attribute, IntegerSetAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(location_attribute, LocationAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(opaque_attribute, OpaqueAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(strided_layout_attribute, StridedLayoutAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(string_attribute, StringAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(symbol_ref_attribute, SymbolRefAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(symbol_visibility_attribute, SymbolVisibilityAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(type_attribute, TypeAttributeRef<'c, 't>);
    mlir_operation_builtin_attribute!(unit_attribute, UnitAttributeRef<'c, 't>);

    /// Returns a named dense 32-bit integer array attribute element as a `usize`.
    ///
    /// This is primarily useful for operation segment-size attributes, whose MLIR representation is a dense i32
    /// array but whose values are naturally used as Rust indices and counts.
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of the operation attribute to retrieve.
    ///   - `index`: Index of the attribute element to retrieve.
    fn dense_integer_32_array_attribute_usize_value<N: AsRef<str>>(
        &self,
        name: N,
        index: usize,
    ) -> Result<usize, Error> {
        let name = name.as_ref();
        let attribute = self.dense_integer_32_array_attribute(name)?;
        if index >= attribute.len() {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                name,
                self.name().as_str().unwrap_or("<unknown>"),
            )));
        }
        usize::try_from(attribute.value(index)).map_err(|_| {
            Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                name,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the flat operand/result index range represented by one segment-size entry.
    ///
    /// This helper interprets the named [`DenseInteger32ArrayAttributeRef`] as an MLIR segment-size attribute, such as
    /// `operandSegmentSizes` or `operand_segment_sizes`. In that convention, operations store multiple variadic operand
    /// or result groups as one flat list, and the dense i32 array stores the length of each consecutive group.
    ///
    /// For example, a segment-size attribute with values `[2, 3, 1]` describes three consecutive groups
    /// in the flat list:
    ///
    /// ```text
    /// segment 0: size 2 -> range 0..2
    /// segment 1: size 3 -> range 2..5
    /// segment 2: size 1 -> range 5..6
    /// ```
    ///
    /// Therefore, requesting segment `1` returns `2..5`. The returned range indexes the operation's flat operand or
    /// result list; it is not a range over the dense array attribute values themselves.
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of the operation attribute to retrieve.
    ///   - `index`: Index of the segment to retrieve.
    fn dense_integer_32_array_attribute_segment_range<N: AsRef<str>>(
        &self,
        name: N,
        index: usize,
    ) -> Result<Range<usize>, Error> {
        let name = name.as_ref();
        let operation_name = self.name().as_str().unwrap_or("<unknown>").to_string();
        let attribute = self.dense_integer_32_array_attribute(name)?;
        if index >= attribute.len() {
            return Err(Error::invalid_argument(format!("invalid `{name}` attribute in `{operation_name}`")));
        }
        let mut start = 0usize;
        for segment in 0..index {
            start = start
                .checked_add(usize::try_from(attribute.value(segment)).map_err(|_| {
                    Error::invalid_argument(format!("invalid `{name}` attribute in `{operation_name}`"))
                })?)
                .ok_or_else(|| Error::invalid_argument(format!("invalid `{name}` attribute in `{operation_name}`")))?;
        }
        let size = usize::try_from(attribute.value(index))
            .map_err(|_| Error::invalid_argument(format!("invalid `{name}` attribute in `{operation_name}`")))?;
        let end = start
            .checked_add(size)
            .ok_or_else(|| Error::invalid_argument(format!("invalid `{name}` attribute in `{operation_name}`")))?;
        Ok(start..end)
    }

    /// Sets an attribute of this [`Operation`] with the provided name to the provided value.
    ///
    /// It is recommended to instead use [`Operation::set_inherent_attribute`] or
    /// [`Operation::set_discardable_attribute`]. Refer to the documentation of [`Operation::attributes`] for
    /// information on the distinction between inherent and discardable attributes.
    fn set_attribute<N: AsRef<str>, A: Attribute<'c, 't>>(&mut self, name: N, value: A) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            mlirOperationSetAttributeByName(
                self.to_c_api(),
                StringRef::from(name.as_ref()).to_c_api(),
                value.to_c_api(),
            )
        }
    }

    /// Removes the attribute of this [`Operation`] with the provided name, returning `true` if the attribute was
    /// removed successfully and `false` otherwise (e.g., if no such attribute could be found).
    ///
    /// It is recommended to instead use [`Operation::remove_discardable_attribute`] as inherent attributes
    /// cannot be removed. Refer to the documentation of [`Operation::attributes`] for information on the
    /// distinction between inherent and discardable attributes.
    fn remove_attribute<N: AsRef<str>>(&mut self, name: N) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe { mlirOperationRemoveAttributeByName(self.to_c_api(), StringRef::from(name.as_ref()).to_c_api()) }
    }

    /// Returns the number of operands of this [`Operation`].
    fn operand_count(&self) -> usize {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirOperationGetNumOperands(self.to_c_api()).cast_unsigned() }
    }

    /// Returns an [`Iterator`] over the operands of this [`Operation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`]
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn operands(&self) -> impl Iterator<Item = Result<OperandRef<'o, 'c, 't>, Error>> {
        let operand_count = self.operand_count();
        (0..operand_count).map(|index| self.operand(index))
    }

    /// Returns the operand at the `index`-pth position in the operands list of this [`Operation`],
    /// or an [`Error`] if `index` is out of bounds.
    fn operand(&self, index: usize) -> Result<OperandRef<'o, 'c, 't>, Error> {
        let operand_count = self.operand_count();
        if index >= operand_count {
            return Err(Error::invalid_argument(format!(
                "operation operand index {index} is out of bounds for length {operand_count}",
            )));
        }

        // The following context borrow ensures that access to the underlying MLIR data structures is done safely
        // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
        // to MLIR internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe {
            OperandRef::from_c_api(mlirOperationGetOpOperand(self.to_c_api(), index.cast_signed()), self.context())
        }
    }

    /// Returns an [`Iterator`] over the operand [`Value`]s of this [`Operation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`]
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn operand_values(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        let operand_count = self.operand_count();
        (0..operand_count).map(|index| self.operand_value(index))
    }

    /// Returns the operand [`Value`] at the `index`-pth position in the operands list of this [`Operation`],
    /// or an [`Error`] if `index` is out of bounds.
    fn operand_value(&self, index: usize) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let operand_count = self.operand_count();
        if index >= operand_count {
            return Err(Error::invalid_argument(format!(
                "operation operand index {index} is out of bounds for length {operand_count}",
            )));
        }

        // The following context borrow ensures that access to the underlying MLIR data structures is done safely
        // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
        // to MLIR internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { ValueRef::from_c_api(mlirOperationGetOperand(self.to_c_api(), index.cast_signed()), self.context()) }
    }

    /// Returns an [`Iterator`] over the [`Type`](crate::Type)s of the [`Operation::operands`] of this [`Operation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`]
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn operand_types(&self) -> impl Iterator<Item = Result<TypeRef<'c, 't>, Error>> {
        self.operand_values().map(|operand| operand.and_then(|operand| operand.r#type()))
    }

    /// Returns the [`Type`](crate::Type) of the operand at the `index`-pth position in the operands list of this
    /// [`Operation`], or an [`Error`] if `index` is out of bounds.
    fn operand_type(&self, index: usize) -> Result<TypeRef<'c, 't>, Error> {
        self.operand_value(index)?.r#type()
    }

    /// Replaces the operand at the `index`-pth position in the operands list of this [`Operation`], with the provided
    /// [`Value`]. Returns `true` if the operation was successful and `false` otherwise (e.g., if the index was out
    /// of bounds).
    ///
    /// Note that this function is marked as _unsafe_ because if the provided [`Value`] does not _dominate_ this
    /// [`Operation`] according to MLIR's dominance rules (i.e., it is not defined before/above it in the current
    /// control flow of the program), then calling this function results in undefined behavior.
    unsafe fn replace_operand<V: Value<'o, 'c, 't>>(&mut self, index: usize, value: V) -> bool {
        if index >= self.operand_count() {
            false
        } else {
            // The following context borrow ensures that access to the underlying MLIR data structures is done safely
            // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
            // to MLIR internals that we have when working with the MLIR C API.
            let _guard = self.context().borrow_mut();
            unsafe { mlirOperationSetOperand(self.to_c_api(), index.cast_signed(), value.to_c_api()) };
            true
        }
    }

    /// Replaces the operands of this [`Operation`] with the provided [`Value`]s. Returns `true` if the operation was
    /// successful and `false` otherwise (e.g., if the number of the provided operands does not match the number
    /// of operands of this operation).
    ///
    /// Note that this function is marked as _unsafe_ because if the provided [`Value`]s do not all _dominate_ this
    /// [`Operation`] according to MLIR's dominance rules (i.e., they are not all defined before/above it in the current
    /// control flow of the program), then calling this function results in undefined behavior.
    unsafe fn replace_operands<'v: 'o, V: Value<'o, 'c, 't>>(&mut self, operands: &[V]) -> bool {
        if operands.len() != self.operand_count() {
            false
        } else {
            // The following context borrow ensures that access to the underlying MLIR data structures is done safely
            // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
            // to MLIR internals that we have when working with the MLIR C API.
            let _guard = self.context().borrow_mut();
            unsafe {
                let operands = operands.iter().map(|operand| operand.to_c_api()).collect::<Vec<_>>();
                mlirOperationSetOperands(self.to_c_api(), operands.len().cast_signed(), operands.as_ptr() as *const _)
            };
            true
        }
    }

    /// Replaces all uses of the `target` [`Value`] inside this [`Operation`] with the provided `replacement`.
    ///
    /// Note that this function is marked as _unsafe_ because if the provided `replacement` does not _dominate_ this
    /// [`Operation`] according to MLIR's dominance rules (i.e., it is not defined before/above it in the current
    /// control flow of the program), then calling this function results in undefined behavior.
    unsafe fn replace_uses_of_with<'a, 'b, A: Value<'a, 'c, 't>, B: Value<'b, 'c, 't>>(
        &mut self,
        target: A,
        replacement: B,
    ) where
        'c: 'a + 'b,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe { mlirOperationReplaceUsesOfWith(self.to_c_api(), target.to_c_api(), replacement.to_c_api()) }
    }

    /// Returns the number of results of this [`Operation`].
    fn result_count(&self) -> usize {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirOperationGetNumResults(self.to_c_api()).cast_unsigned() }
    }

    /// Returns an [`Iterator`] over the results of this [`Operation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`]
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn results(&self) -> impl Iterator<Item = Result<OperationResultRef<'o, 'c, 't>, Error>> {
        let result_count = self.result_count();
        (0..result_count).map(|index| self.result(index))
    }

    /// Returns the result at the `index`-pth position in the results list of this [`Operation`],
    /// or an [`Error`] if `index` is out of bounds.
    fn result(&self, index: usize) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        let result_count = self.result_count();
        if index >= result_count {
            return Err(Error::invalid_argument(format!(
                "operation result index {index} is out of bounds for length {result_count}",
            )));
        }

        // The following context borrow ensures that access to the underlying MLIR data structures is done safely
        // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
        // to MLIR internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe {
            OperationResultRef::from_c_api(mlirOperationGetResult(self.to_c_api(), index.cast_signed()), self.context())
        }
    }

    /// Returns an [`Iterator`] over the [`Type`](crate::Type)s of the [`Operation::results`] of this [`Operation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`]
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn result_types(&self) -> impl Iterator<Item = Result<TypeRef<'c, 't>, Error>> {
        self.results().map(|result| result.and_then(|result| result.r#type()))
    }

    /// Returns the [`Type`](crate::Type) of the result at the `index`-pth position in the results list of this
    /// [`Operation`], or an [`Error`] if `index` is out of bounds.
    fn result_type(&self, index: usize) -> Result<TypeRef<'c, 't>, Error> {
        self.result(index)?.r#type()
    }

    /// Returns `true` if this [`Operation`] is empty (i.e., if it contains no [`Region`](crate::Region)s).
    fn is_empty(&self) -> bool {
        self.region_count() == 0
    }

    /// Returns the number of [`Region`](crate::Region)s of this [`Operation`].
    fn region_count(&self) -> usize {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirOperationGetNumRegions(self.to_c_api()).cast_unsigned() }
    }

    /// Returns an [`Iterator`] over the [`Region`](crate::Region)s of this [`Operation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`]
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn regions(&self) -> impl Iterator<Item = Result<RegionRef<'o, 'c, 't>, Error>> {
        let region_count = self.region_count();
        (0..region_count).map(|index| self.region(index))
    }

    /// Returns the [`Region`](crate::Region) at the `index`-pth position in the [`Region`](crate::Region)s list
    /// of this [`Operation`], or an [`Error`] if `index` is out of bounds.
    fn region(&self, index: usize) -> Result<RegionRef<'o, 'c, 't>, Error> {
        let region_count = self.region_count();
        if index >= region_count {
            return Err(Error::invalid_argument(format!(
                "operation region index {index} is out of bounds for length {region_count}",
            )));
        }

        // The following context borrow ensures that access to the underlying MLIR data structures is done safely
        // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
        // to MLIR internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { RegionRef::from_c_api(mlirOperationGetRegion(self.to_c_api(), index.cast_signed()), self.context()) }
    }

    /// Returns the number of successor [`Block`]s of this [`Operation`]. Refer to [`Block::successors`]
    /// for information on how successors are defined.
    fn successor_count(&self) -> usize {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirOperationGetNumSuccessors(self.to_c_api()).cast_unsigned() }
    }

    /// Returns an [`Iterator`] over the successor [`Block`]s of this [`Operation`]. Refer to [`Block::successors`]
    /// for information on how successors are defined.
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`]
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn successors(&self) -> impl Iterator<Item = Result<BlockRef<'o, 'c, 't>, Error>> {
        let successor_count = self.successor_count();
        (0..successor_count).map(|index| self.successor(index))
    }

    /// Returns the successor [`Block`] at the `index`-pth position in the successors list of this [`Operation`],
    /// or an [`Error`] if `index` is out of bounds. Refer to [`Block::successors`] for information
    /// on how successors are defined.
    fn successor(&self, index: usize) -> Result<BlockRef<'o, 'c, 't>, Error> {
        let successor_count = self.successor_count();
        if index >= successor_count {
            return Err(Error::invalid_argument(format!(
                "operation successor index {index} is out of bounds for length {successor_count}",
            )));
        }

        // The following context borrow ensures that access to the underlying MLIR data structures is done safely
        // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
        // to MLIR internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { BlockRef::from_c_api(mlirOperationGetSuccessor(self.to_c_api(), index.cast_signed()), self.context()) }
    }

    /// Replaces the successor at the `index`-pth position in the successors list of this [`Operation`], with the
    /// provided [`Block`]. Returns `true` if the operation was successful and `false` otherwise (e.g., if the index
    /// was out of bounds).
    fn replace_successor<B: Block<'o, 'c, 't>>(&mut self, index: usize, block: &B) -> bool {
        if index >= self.successor_count() {
            false
        } else {
            // The following context borrow ensures that access to the underlying MLIR data structures is done safely
            // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
            // to MLIR internals that we have when working with the MLIR C API.
            let _guard = self.context().borrow_mut();
            unsafe { mlirOperationSetSuccessor(self.to_c_api(), index.cast_signed(), block.to_c_api()) };
            true
        }
    }

    /// Returns a reference to the parent [`Block`] of this [`Operation`] (i.e., the [`Block`] that owns this
    /// operation), if one exists (i.e., if this is not a [`DetachedOperation`] or a reference to a detached
    /// operation).
    fn parent_block(&self) -> Result<Option<BlockRef<'o, 'c, 't>>, Error> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        let handle = unsafe { mlirOperationGetBlock(self.to_c_api()) };
        if handle.ptr.is_null() { Ok(None) } else { unsafe { BlockRef::from_c_api(handle, self.context()).map(Some) } }
    }

    /// Returns a reference to the parent [`Operation`] of this [`Operation`] (i.e., the [`Operation`] that owns this
    /// operation), if one exists (i.e., if this is not a [`DetachedOperation`] or a reference to a detached
    /// operation).
    fn parent_operation(&self) -> Result<Option<OperationRef<'o, 'c, 't>>, Error> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        let handle = unsafe { mlirOperationGetParentOperation(self.to_c_api()) };
        if handle.ptr.is_null() {
            Ok(None)
        } else {
            unsafe { OperationRef::from_c_api(handle, self.context()).map(Some) }
        }
    }

    /// Returns `true` if this operation appears before `other` in the parent [`Block`] of this operation (assuming
    /// that `other` belongs to the same [`Block`]; this function will return `false` if that is not the case).
    ///
    /// Note that this function has an average complexity of `O(1)` but in the worst case it may take `O(N)` where `N`
    /// is the number of [`Operation`]s in the parent [`Block`].
    fn is_before_in_block<'b, O: OpRef<'b, 'c, 't>>(&self, other: &O) -> bool
    where
        'c: 'b,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirOperationIsBeforeInBlock(self.to_c_api(), other.to_c_api()) }
    }

    /// Moves this [`Operation`] immediately after the provided `other` operation in its parent [`Block`].
    ///
    /// This function is marked as unsafe because it cannot protect against memory issues arising from calling this
    /// function on an [`OperationRef`] that is a reference to a [`DetachedOperation`]. This would be problematic
    /// because [`std::mem::forget`] will not be called on the underlying [`DetachedOperation`], meaning that it may
    /// be dropped while the parent [`Block`] of `other` is still alive.
    unsafe fn move_after<'b, O: OpRef<'b, 'c, 't>>(self, other: &O) -> OperationRef<'b, 'c, 't>
    where
        'c: 'b,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            // We forget `self` and return a new `OperationRef` to make sure that ownership is transferred
            // correctly if `self` is a `DetachedOperation`.
            let context = self.context();
            let handle = self.to_c_api();
            mlirOperationMoveAfter(handle, other.to_c_api());
            if !handle.ptr.is_null() {
                std::mem::forget(self);
            }
            OperationRef { handle, context, owner: PhantomData }
        }
    }

    /// Moves this [`Operation`] immediately before the provided `other` operation in its parent [`Block`].
    ///
    /// This function is marked as unsafe because it cannot protect against memory issues arising from calling this
    /// function on an [`OperationRef`] that is a reference to a [`DetachedOperation`]. This would be problematic
    /// because [`std::mem::forget`] will not be called on the underlying [`DetachedOperation`], meaning that it may
    /// be dropped while the parent [`Block`] of `other` is still alive.
    unsafe fn move_before<'b, O: OpRef<'b, 'c, 't>>(self, other: &O) -> OperationRef<'b, 'c, 't>
    where
        'c: 'b,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            // We forget `self` and return a new `OperationRef` to make sure that ownership is transferred
            // correctly if `self` is a `DetachedOperation`.
            let context = self.context();
            let handle = self.to_c_api();
            mlirOperationMoveBefore(handle, other.to_c_api());
            if !handle.ptr.is_null() {
                std::mem::forget(self);
            }
            OperationRef { handle, context, owner: PhantomData }
        }
    }

    /// Performs a walk over this [`Operation`] (i.e., itself and all of its nested operations) in the specified
    /// [`WalkOrder`], invoking `callback` on each operation it visits. The traversal is also controlled by the
    /// result of each `callback` invocation as it can determine whether to advance to the next operation, skip
    /// the next operation, or completely interrupt the walk.
    ///
    /// Note that this function does not support callbacks that mutate the associated [`Context`] and if such callbacks
    /// are used, they will result in runtime panics.
    fn walk<F: FnMut(OperationRef<'o, 'c, 't>) -> WalkResult>(&self, order: WalkOrder, mut callback: F) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();

        unsafe extern "C" fn c_api_callback<'o, 'c: 'o, 't: 'c, F: FnMut(OperationRef<'o, 'c, 't>) -> WalkResult>(
            operation: MlirOperation,
            data: *mut std::ffi::c_void,
        ) -> MlirWalkResult {
            unsafe {
                let data = data as *mut (&mut F, &'c Context<'t>);
                let (ref mut callback, context) = *data;
                OperationRef::from_c_api(operation, context)
                    .map(|operation| (callback)(operation).to_c_api())
                    .unwrap_or(MlirWalkResult_MlirWalkResultInterrupt)
            }
        }

        unsafe {
            mlirOperationWalk(
                self.to_c_api(),
                Some(c_api_callback::<'o, 'c, 't, F>),
                &mut (&mut callback, self.context()) as *mut _ as *mut _,
                order.to_c_api(),
            );
        }
    }

    /// Returns the bytecode representation of this [`Operation`] using the default [`BytecodeWriterConfiguration`].
    fn bytecode(&self) -> Vec<u8> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        let mut data = Vec::new();
        unsafe {
            mlirOperationWriteBytecode(
                self.to_c_api(),
                Some(write_to_bytes_callback),
                &mut data as *mut _ as *mut std::ffi::c_void,
            );
        }
        data
    }

    /// Returns the bytecode representation of this [`Operation`] using the provided [`BytecodeWriterConfiguration`].
    /// Note that if the bytecode generation fails for the provided configuration, then this function will
    /// return `Ok(None)`.
    fn bytecode_with_configuration(&self, configuration: &BytecodeWriterConfiguration) -> Option<Vec<u8>> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe {
            let configuration_handle = configuration.handle();
            let mut data = Vec::new();
            let result = mlirOperationWriteBytecodeWithConfig(
                self.to_c_api(),
                configuration_handle.to_c_api(),
                Some(write_to_bytes_callback),
                &mut data as *mut _ as *mut std::ffi::c_void,
            );
            if LogicalResult::from_c_api(result).is_failure() { None } else { Some(data) }
        }
    }

    /// Returns the bytecode representation for this [`Operation`] using the specified version. This function calls
    /// [`Operation::bytecode_for_version`] internally.
    fn bytecode_for_version(&self, version: u64) -> Option<Vec<u8>> {
        self.bytecode_with_configuration(&BytecodeWriterConfiguration { version: Some(version) })
    }

    /// Renders this [`Operation`] as a string using the provided [`OperationPrintingFlags`].
    fn to_string_with_flags(&self, flags: OperationPrintingFlags) -> Result<String, std::str::Utf8Error> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        let mut data = (String::new(), Ok(()));
        unsafe {
            let flags_handle = flags.handle();
            mlirOperationPrintWithFlags(
                self.to_c_api(),
                flags_handle.to_c_api(),
                Some(write_to_string_callback),
                &mut data as *mut _ as *mut std::ffi::c_void,
            );
        }
        data.1.map(|_| data.0)
    }

    /// Renders this [`Operation`] as a string using the provided [`AsmState`] with controls the rendering behavior
    /// as well as the caching of computed names.
    fn to_string_with_state(&self, state: AsmState) -> Result<String, std::str::Utf8Error> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        let mut data = (String::new(), Ok(()));
        unsafe {
            mlirOperationPrintWithState(
                self.to_c_api(),
                state.to_c_api(),
                Some(write_to_string_callback),
                &mut data as *mut _ as *mut std::ffi::c_void,
            );
        }
        data.1.map(|_| data.0)
    }

    /// Verifies this [`Operation`] (as in, checks if it is well-defined) and returns `true` if the verification passes.
    fn verify(&self) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirOperationVerify(self.to_c_api()) }
    }

    /// Dumps this [`Operation`] to the standard error stream.
    fn dump(&self) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirOperationDump(self.to_c_api()) }
    }
}

/// Trait used to represent detached (i.e., owned) [`Operation`]s.
pub trait DetachedOp<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Attempts to cast this [`DetachedOp`] to the provided [`DetachedOp`] type. This function is unsafe because
    /// there is no guaranteed/safe way to ensure that the desired cast is valid. Therefore, attempting an invalid
    /// cast can result in undefined behavior and so this function needs to be used with care.
    unsafe fn cast<O: DetachedOp<'o, 'c, 't>>(self) -> Option<O> {
        let operation = unsafe { O::from_c_api(self.to_c_api(), self.context()).ok() };
        if operation.is_some() {
            std::mem::forget(self);
        }
        operation
    }
}

/// Trait used to represent non-owning references to [`Operation`]s.
pub trait OpRef<'o, 'c: 'o, 't: 'c>: Copy + Operation<'o, 'c, 't> {
    /// Tries to cast this [`OpRef`] to an instance of `O` (e.g., an instance of [`OperationRef`]). If this
    /// is not an instance of the specified [`OpRef`] type, this function will return [`None`].
    unsafe fn cast<O: OpRef<'o, 'c, 't>>(&self) -> Option<O> {
        unsafe { O::from_c_api(self.to_c_api(), self.context()).ok() }
    }
}

/// [`Operation`] that is not part of an MLIR program (i.e., it is "detached") and is not owned by a [`Block`] in the
/// current [`Context`]. [`DetachedOperation`]s can be added to [`Block`]s (e.g., using [`Block::append_operation`]),
/// handing off ownership to the respective [`Block`]. While it is not strictly necessary that a [`DetachedOperation`]
/// keeps a pointer to an MLIR [`Context`] (and its lifetimes), this structure does keep that pointer around (and its
/// lifetimes) as a means to provide more safety when accessing and potentially mutating objects nested inside
/// [`DetachedOperation`]s. Note that this is technically also more "correct" in that there are objects referenced by
/// even [`DetachedOperation`]s that are owned and managed by the associated [`Context`] (e.g., [`Location`]s and
/// [`Type`](crate::Type)s).
pub struct DetachedOperation<'c, 't: 'c> {
    /// Handle that represents this [`Operation`] in the MLIR C API.
    handle: MlirOperation,

    /// [`Context`] associated with this [`Operation`].
    context: &'c Context<'t>,
}

impl<'o, 'c: 'o, 't: 'c> Operation<'o, 'c, 't> for DetachedOperation<'c, 't> {
    unsafe fn from_c_api(handle: MlirOperation, context: &'c Context<'t>) -> Result<Self, Error> {
        if handle.ptr.is_null() {
            Err(Error::internal("expected non-null MLIR operation handle"))
        } else {
            Ok(Self { handle, context })
        }
    }

    unsafe fn to_c_api(&self) -> MlirOperation {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

impl<'o, 'c: 'o, 't: 'c> DetachedOp<'o, 'c, 't> for DetachedOperation<'c, 't> {}

impl Clone for DetachedOperation<'_, '_> {
    fn clone(&self) -> Self {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context.borrow();
        Self { handle: unsafe { mlirOperationClone(self.handle) }, context: self.context }
    }
}

impl<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>> PartialEq<O> for DetachedOperation<'c, 't> {
    fn eq(&self, other: &O) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context.borrow();
        // Note that this function only checks for whether the two operation handles point to the same underlying
        // operation. It does not perform a deep comparison of the contents of these operations.
        unsafe { mlirOperationEqual(self.handle, other.to_c_api()) }
    }
}

impl Eq for DetachedOperation<'_, '_> {}

impl Hash for DetachedOperation<'_, '_> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        unsafe { mlirOperationHashValue(self.handle).hash(hasher) }
    }
}

impl Display for DetachedOperation<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context.borrow();
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirOperationPrint(
                self.to_c_api(),
                Some(write_to_formatter_callback),
                &mut data as *mut _ as *mut std::ffi::c_void,
            );
        }
        data.1
    }
}

impl Debug for DetachedOperation<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "DetachedOperation[{self}]")
    }
}

impl Drop for DetachedOperation<'_, '_> {
    fn drop(&mut self) {
        if !self.handle.ptr.is_null() {
            // The following context borrow ensures that access to the underlying MLIR data structures is done safely
            // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
            // to MLIR internals that we have when working with the MLIR C API.
            let _guard = self.context.borrow_mut();
            unsafe { mlirOperationDestroy(self.handle) }
        }
    }
}

impl<'t> Context<'t> {
    /// Parses a [`DetachedOperation`] from the provided string representation.
    ///
    /// Returns an [`Error`] if MLIR fails to parse the provided string into an [`Operation`] (this function will also
    /// emit diagnostics if that happens). The provided `filename` is used to create a
    /// [`FileLocationRef`](crate::FileLocationRef) that will be used as the location of the resulting [`Operation`].
    pub fn parse_operation<'o, 'c: 'o>(
        &'c self,
        source: &str,
        filename: &str,
    ) -> Result<DetachedOperation<'c, 't>, Error> {
        self.parse_operation_from_bytes(source.as_bytes(), filename)
    }

    /// Parses a [`DetachedOperation`] directly from the provided bytes. The bytes may contain either textual MLIR or
    /// MLIR bytecode and are passed to the native parser without UTF-8 validation or `nul` termination.
    ///
    /// Returns an [`Error`] if MLIR fails to parse the provided bytes. MLIR will also emit structured diagnostics to
    /// handlers attached with [`Context::attach_diagnostics_handler`]. The provided `filename` is used as the source
    /// name for those diagnostics and for parsed locations.
    pub fn parse_operation_from_bytes<'o, 'c: 'o, B: AsRef<[u8]>>(
        &'c self,
        source: B,
        filename: &str,
    ) -> Result<DetachedOperation<'c, 't>, Error> {
        unsafe {
            let handle = mlirOperationCreateParse(
                // The following context borrow ensures that access to the underlying MLIR data structures is done
                // safely from Rust. It is maybe more conservative than would be ideal, but that is due to the
                // limited exposure to MLIR internals that we have when working with the MLIR C API.
                *self.handle.borrow(),
                StringRef::from(source.as_ref()).to_c_api(),
                StringRef::from(filename).to_c_api(),
            );
            if handle.ptr.is_null() {
                Err(Error::parsing_error(format!("failed to parse MLIR operation from `{filename}`")))
            } else {
                DetachedOperation::from_c_api(handle, self)
            }
        }
    }
}

/// Reference to an MLIR [`Operation`] that is owned by a [`Block`].
///
/// Note that there are multiple separate lifetime parameters: one for the lifetime of this [`Operation`] reference,
/// `'o`, one for the [`Context`] which is associated with it, `'c`, and one for the lifetime of the thread pool used
/// by that [`Context`], `'t`.
#[derive(Copy, Clone)]
pub struct OperationRef<'o, 'c: 'o, 't: 'c> {
    /// Handle that represents this [`Operation`] reference in the MLIR C API.
    handle: MlirOperation,

    /// [`Context`] associated with this [`Operation`] reference.
    context: &'c Context<'t>,

    /// [`PhantomData`] used to track the lifetime of the [`Block`] that owns the underlying [`Operation`].
    owner: PhantomData<&'o ()>,
}

impl<'r, 'o: 'r, 'c: 'o, 't: 'c> Operation<'r, 'c, 't> for OperationRef<'o, 'c, 't> {
    unsafe fn from_c_api(handle: MlirOperation, context: &'c Context<'t>) -> Result<Self, Error> {
        if handle.ptr.is_null() {
            Err(Error::internal("expected non-null MLIR operation handle"))
        } else {
            Ok(Self { handle, context, owner: PhantomData })
        }
    }

    unsafe fn to_c_api(&self) -> MlirOperation {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

impl<'r, 'o: 'r, 'c: 'o, 't: 'c> OpRef<'r, 'c, 't> for OperationRef<'o, 'c, 't> {}

impl<'r, 'o, 'c: 'r, 't: 'c, O: Operation<'r, 'c, 't>> PartialEq<O> for OperationRef<'o, 'c, 't> {
    fn eq(&self, other: &O) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context.borrow();
        // Note that this function only checks for whether the two operation handles point to the same underlying
        // operation. It does not perform a deep comparison of the contents of these operations.
        unsafe { mlirOperationEqual(self.handle, other.to_c_api()) }
    }
}

impl Eq for OperationRef<'_, '_, '_> {}

impl Hash for OperationRef<'_, '_, '_> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        unsafe { mlirOperationHashValue(self.handle).hash(hasher) }
    }
}

impl Display for OperationRef<'_, '_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context.borrow();
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirOperationPrint(
                self.to_c_api(),
                Some(write_to_formatter_callback),
                &mut data as *mut _ as *mut std::ffi::c_void,
            );
        }
        data.1
    }
}

impl Debug for OperationRef<'_, '_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "OperationRef[{self}]")
    }
}

/// Traversal order when performing a walk over [`Operation`]s.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum WalkOrder {
    /// Each [`Operation`] will be visited before its nested regions are visited.
    PreOrder,

    /// Each [`Operation`] will be visited after its nested regions are visited.
    PostOrder,
}

impl WalkOrder {
    /// Returns the [`MlirWalkOrder`] that corresponds to this [`WalkOrder`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirWalkOrder {
        match self {
            WalkOrder::PreOrder => MlirWalkOrder_MlirWalkPreOrder,
            WalkOrder::PostOrder => MlirWalkOrder_MlirWalkPostOrder,
        }
    }
}

/// Result returned by the callback that is used when performing walks over [`Operation`]s
/// and which determines the next action to take in the current walk.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum WalkResult {
    /// The traversal should continue with the step in the walk.
    Advance,

    /// The traversal should terminate without continuing with the rest of the walk.
    Interrupt,

    /// The traversal should skip the current [`Operation`]'s children and move directly to its siblings
    /// (or to its parent's siblings if it does not have any siblings, etc.).
    Skip,
}

impl WalkResult {
    /// Returns the [`MlirWalkResult`] that corresponds to this [`WalkResult`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirWalkResult {
        match self {
            WalkResult::Advance => MlirWalkResult_MlirWalkResultAdvance,
            WalkResult::Interrupt => MlirWalkResult_MlirWalkResultInterrupt,
            WalkResult::Skip => MlirWalkResult_MlirWalkResultSkip,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{
        Block, Context, DetachedModuleOperation, DiagnosticSeverity, DialectHandle, OperationBuilder, Region, Size,
        SymbolVisibility, Type, Value, ValueRef,
    };

    use super::*;

    #[test]
    fn test_operation_construction() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();

        // Test using a simple unregistered operation that has no type ID.
        let mut op = OperationBuilder::new("foo", location).build().unwrap();
        assert_eq!(op, op.as_ref());
        assert!(op.type_id().unwrap().is_none());
        assert_eq!(op.location().unwrap(), location);
        op.set_location(context.file_location("test.mlir", 4, 2));
        assert_eq!(op.location().unwrap(), context.file_location("test.mlir", 4, 2));
        assert_eq!(op.name(), context.identifier("foo"));

        // Check that registered operations have type IDs.
        let mut block = context.block_with_no_arguments();
        block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
        let op = func::func("test_func", func::FuncAttributes::default(), block.try_into().unwrap(), location).unwrap();
        assert!(op.type_id().unwrap().is_some());

        // Test a C API-related edge case.
        let op = DetachedOperation { handle: MlirOperation { ptr: std::ptr::null_mut() }, context: &context };
        assert!(unsafe { op.cast::<DetachedModuleOperation>() }.is_none());
    }

    #[test]
    fn test_operation_inherent_attributes() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
        let mut op =
            func::func("test_func", func::FuncAttributes::default(), block.try_into().unwrap(), location).unwrap();
        assert!(op.inherent_attribute_count() > 0);
        assert!(op.has_inherent_attribute("sym_name"));
        assert_eq!(op.inherent_attribute("sym_name").unwrap(), Some(context.string_attribute("test_func").as_ref()));
        op.set_inherent_attribute("sym_name", context.string_attribute("modified"));
        assert_eq!(op.inherent_attribute("sym_name").unwrap(), Some(context.string_attribute("modified").as_ref()));
    }

    #[test]
    fn test_operation_discardable_attributes() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();

        // Create a simple operation that has no attributes to start with.
        let mut op = OperationBuilder::new("test.op", location).build().unwrap();
        assert_eq!(op.discardable_attribute_count(), 0);
        assert!(!op.has_discardable_attribute("custom"));

        // Add a discardable attribute.
        op.set_discardable_attribute("custom", context.string_attribute("value"));
        assert_eq!(op.discardable_attribute_count(), 1);
        assert!(op.has_discardable_attribute("custom"));
        assert_eq!(op.discardable_attribute("custom").unwrap(), Some(context.string_attribute("value").as_ref()));
        let attributes = op.discardable_attributes().collect::<Vec<_>>();
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0].name(), context.identifier("custom"));

        // Remove a discardable attribute.
        assert!(op.remove_discardable_attribute("custom"));
        assert_eq!(op.discardable_attribute_count(), 0);
        assert!(!op.remove_discardable_attribute("custom"));

        // Try also removing it by setting it to the null attribute.
        op.set_discardable_attribute("custom", context.string_attribute("value"));
        assert_eq!(op.discardable_attribute_count(), 1);
        assert!(op.has_discardable_attribute("custom"));
        op.set_discardable_attribute("custom", context.null_attribute());
        assert_eq!(op.discardable_attribute_count(), 0);
        assert!(!op.has_discardable_attribute("custom"));
    }

    #[test]
    fn test_operation_attributes() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let mut op = OperationBuilder::new("foo", location)
            .add_attribute("foo", context.string_attribute("bar"))
            .build()
            .unwrap();
        assert!(op.attribute("foo").unwrap().is_some());
        assert_eq!(op.attribute("foo").unwrap().map(|attribute| attribute.to_string()), Some("\"bar\"".into()));
        assert!(op.remove_attribute("foo"));
        assert!(!op.remove_attribute("foo"));
        op.set_attribute("foo", context.string_attribute("foo"));
        assert_eq!(op.attribute("foo").unwrap().map(|attribute| attribute.to_string()), Some("\"foo\"".into()));
        let attribute = op.attributes().next().unwrap();
        assert_eq!(attribute.name(), context.identifier("foo"));
        assert_eq!(attribute.attribute().unwrap(), context.string_attribute("foo"));

        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, location).unwrap();
        let float_tensor_type =
            context.tensor_type(context.float32_type(), &[Size::Static(2)], None, location).unwrap();

        let affine_map_attribute = context.affine_map_attribute(context.empty_affine_map());
        op.set_attribute("affine_map", affine_map_attribute);
        assert_eq!(op.affine_map_attribute("affine_map").unwrap(), affine_map_attribute);

        let array_attribute =
            context.array_attribute(&[context.integer_attribute(i32_type, 1), context.integer_attribute(i32_type, 2)]);
        op.set_attribute("array", array_attribute);
        assert_eq!(op.array_attribute("array").unwrap(), array_attribute);

        let boolean_attribute = context.boolean_attribute(true);
        op.set_attribute("boolean", boolean_attribute);
        assert_eq!(op.boolean_attribute("boolean").unwrap(), boolean_attribute);

        let dense_array_attribute = context.dense_i32_array_attribute(&[1, 2, 3]).unwrap();
        op.set_attribute("dense_array", dense_array_attribute);
        assert_eq!(
            op.dense_array_attribute("dense_array").unwrap(),
            dense_array_attribute.cast::<DenseArrayAttributeRef<'_, '_>>().unwrap()
        );

        let dense_boolean_array_attribute = context.dense_bool_array_attribute(&[true, false]).unwrap();
        op.set_attribute("dense_boolean_array", dense_boolean_array_attribute);
        assert_eq!(op.dense_boolean_array_attribute("dense_boolean_array").unwrap(), dense_boolean_array_attribute);

        let dense_integer_8_array_attribute = context.dense_i8_array_attribute(&[1, 2, 3]).unwrap();
        op.set_attribute("dense_integer_8_array", dense_integer_8_array_attribute);
        assert_eq!(
            op.dense_integer_8_array_attribute("dense_integer_8_array").unwrap(),
            dense_integer_8_array_attribute
        );

        let dense_integer_16_array_attribute = context.dense_i16_array_attribute(&[1, 2, 3]).unwrap();
        op.set_attribute("dense_integer_16_array", dense_integer_16_array_attribute);
        assert_eq!(
            op.dense_integer_16_array_attribute("dense_integer_16_array").unwrap(),
            dense_integer_16_array_attribute
        );

        let dense_integer_32_array_attribute = context.dense_i32_array_attribute(&[1, 2, 3]).unwrap();
        op.set_attribute("dense_integer_32_array", dense_integer_32_array_attribute);
        assert_eq!(
            op.dense_integer_32_array_attribute("dense_integer_32_array").unwrap(),
            dense_integer_32_array_attribute
        );

        let dense_integer_64_array_attribute = context.dense_i64_array_attribute(&[1, 2, 3]).unwrap();
        op.set_attribute("dense_integer_64_array", dense_integer_64_array_attribute);
        assert_eq!(
            op.dense_integer_64_array_attribute("dense_integer_64_array").unwrap(),
            dense_integer_64_array_attribute
        );

        let dense_float_32_array_attribute = context.dense_f32_array_attribute(&[1.0, 2.0, 3.0]).unwrap();
        op.set_attribute("dense_float_32_array", dense_float_32_array_attribute);
        assert_eq!(op.dense_float_32_array_attribute("dense_float_32_array").unwrap(), dense_float_32_array_attribute);

        let dense_float_64_array_attribute = context.dense_f64_array_attribute(&[1.0, 2.0, 3.0]).unwrap();
        op.set_attribute("dense_float_64_array", dense_float_64_array_attribute);
        assert_eq!(op.dense_float_64_array_attribute("dense_float_64_array").unwrap(), dense_float_64_array_attribute);

        let dictionary_attribute = context.dictionary_attribute(&[
            context.named_attribute(context.identifier("value"), context.integer_attribute(i32_type, 42))
        ]);
        op.set_attribute("dictionary", dictionary_attribute);
        assert_eq!(op.dictionary_attribute("dictionary").unwrap(), dictionary_attribute);

        let distinct_attribute = context.distinct_attribute(context.integer_attribute(i32_type, 42));
        op.set_attribute("distinct", distinct_attribute);
        assert_eq!(op.attribute("distinct").unwrap().unwrap(), distinct_attribute.as_ref());
        // The MLIR C API does not expose a distinct-attribute predicate, so this accessor cannot downcast attributes
        // recovered from an operation.
        assert!(op.distinct_attribute("distinct").is_err());

        let dense_integer_elements_attribute = context.dense_i32_elements_attribute(tensor_type, &[1, 2]).unwrap();
        op.set_attribute("dense_integer_elements", dense_integer_elements_attribute);
        assert_eq!(
            op.elements_attribute("dense_integer_elements").unwrap(),
            dense_integer_elements_attribute.cast::<ElementsAttributeRef<'_, '_>>().unwrap()
        );
        assert_eq!(
            op.dense_elements_attribute("dense_integer_elements").unwrap(),
            dense_integer_elements_attribute.cast::<DenseElementsAttributeRef<'_, '_>>().unwrap()
        );
        assert_eq!(
            op.dense_integer_elements_attribute("dense_integer_elements").unwrap(),
            dense_integer_elements_attribute
        );

        let dense_float_elements_attribute =
            context.dense_f32_elements_attribute(float_tensor_type, &[1.0, 2.0]).unwrap();
        op.set_attribute("dense_float_elements", dense_float_elements_attribute);
        assert_eq!(op.dense_float_elements_attribute("dense_float_elements").unwrap(), dense_float_elements_attribute);

        let dense_resource_elements_attribute = context
            .dense_i32_resource_elements_attribute(
                tensor_type,
                StringRef::from("operation_attribute_resource"),
                &[1, 2],
            )
            .unwrap();
        op.set_attribute("dense_resource_elements", dense_resource_elements_attribute);
        assert_eq!(
            op.dense_resource_elements_attribute("dense_resource_elements").unwrap(),
            dense_resource_elements_attribute
        );

        let sparse_indices_type =
            context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let sparse_indices = context.dense_i64_elements_attribute(sparse_indices_type, &[0, 0, 1, 1]).unwrap();
        let sparse_values = context.dense_i32_elements_attribute(tensor_type, &[1, 2]).unwrap();
        let sparse_elements_attribute =
            context.sparse_elements_attribute(tensor_type, sparse_indices, sparse_values).unwrap();
        op.set_attribute("sparse_elements", sparse_elements_attribute);
        assert_eq!(op.sparse_elements_attribute("sparse_elements").unwrap(), sparse_elements_attribute);

        let flat_symbol_ref_attribute = context.flat_symbol_ref_attribute("symbol");
        op.set_attribute("flat_symbol_ref", flat_symbol_ref_attribute);
        assert_eq!(op.flat_symbol_ref_attribute("flat_symbol_ref").unwrap(), flat_symbol_ref_attribute);

        let nested_symbol_ref = context.flat_symbol_ref_attribute("nested");
        let symbol_ref_attribute = context.symbol_ref_attribute("root".into(), &[nested_symbol_ref]);
        op.set_attribute("symbol_ref", symbol_ref_attribute);
        assert_eq!(op.symbol_ref_attribute("symbol_ref").unwrap(), symbol_ref_attribute);

        let float_attribute = context.float_attribute(context.float64_type(), 1.5);
        op.set_attribute("float", float_attribute);
        assert_eq!(op.float_attribute("float").unwrap(), float_attribute);

        let integer_attribute = context.integer_attribute(i32_type, 42);
        op.set_attribute("integer", integer_attribute);
        assert_eq!(op.integer_attribute("integer").unwrap(), integer_attribute);

        let integer_set_attribute = context.integer_set_attribute(context.empty_integer_set(0, 1));
        op.set_attribute("integer_set", integer_set_attribute);
        assert_eq!(op.integer_set_attribute("integer_set").unwrap(), integer_set_attribute);

        let location_attribute = context.location_attribute(location);
        op.set_attribute("location", location_attribute);
        assert_eq!(op.location_attribute("location").unwrap(), location_attribute);

        let opaque_attribute = context.opaque_attribute("test_dialect", "opaque_data", context.index_type());
        op.set_attribute("opaque", opaque_attribute);
        assert_eq!(op.opaque_attribute("opaque").unwrap(), opaque_attribute);

        let strided_layout_attribute = context.strided_layout_attribute(0, &[4, 1]);
        op.set_attribute("strided_layout", strided_layout_attribute);
        assert_eq!(op.strided_layout_attribute("strided_layout").unwrap(), strided_layout_attribute);

        let string_attribute = context.string_attribute("value");
        op.set_attribute("string", string_attribute);
        assert_eq!(op.string_attribute("string").unwrap(), string_attribute);

        let symbol_visibility_attribute = context.symbol_visibility_attribute(SymbolVisibility::Private);
        op.set_attribute("symbol_visibility", symbol_visibility_attribute);
        assert_eq!(op.symbol_visibility_attribute("symbol_visibility").unwrap(), symbol_visibility_attribute);

        let type_attribute = context.type_attribute(context.index_type());
        op.set_attribute("type", type_attribute);
        assert_eq!(op.type_attribute("type").unwrap(), type_attribute);

        let unit_attribute = context.unit_attribute();
        op.set_attribute("unit", unit_attribute);
        assert_eq!(op.unit_attribute("unit").unwrap(), unit_attribute);

        op.set_attribute("segments", context.dense_i32_array_attribute(&[2, 3, 1]).unwrap());
        assert_eq!(op.dense_integer_32_array_attribute_usize_value("segments", 0).unwrap(), 2);
        assert_eq!(op.dense_integer_32_array_attribute_usize_value("segments", 1).unwrap(), 3);
        assert_eq!(op.dense_integer_32_array_attribute_usize_value("segments", 2).unwrap(), 1);
        assert_eq!(op.dense_integer_32_array_attribute_segment_range("segments", 0).unwrap(), 0..2);
        assert_eq!(op.dense_integer_32_array_attribute_segment_range("segments", 1).unwrap(), 2..5);
        assert_eq!(op.dense_integer_32_array_attribute_segment_range("segments", 2).unwrap(), 5..6);
        assert!(op.dense_integer_32_array_attribute_usize_value("segments", 3).is_err());
        assert!(op.dense_integer_32_array_attribute_segment_range("segments", 3).is_err());
        assert!(op.boolean_attribute("missing").is_err());
        assert!(op.string_attribute("boolean").is_err());
    }

    #[test]
    fn test_operation_operands() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let index_type = context.index_type().as_ref();

        // Operation with no operands.
        let op = OperationBuilder::new("foo", location).build().unwrap();
        assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);

        // Operation with three operands.
        let block = context.block(&[(index_type, location)]);
        let argument_0 = block.argument(0).unwrap().as_ref();
        let op = OperationBuilder::new("foo", context.unknown_location())
            .add_operand(argument_0)
            .add_operand(argument_0)
            .add_operand(argument_0)
            .build()
            .unwrap();
        assert_eq!(op.operand(0).unwrap().value().unwrap(), argument_0);
        assert_eq!(op.operand(1).unwrap().value().unwrap(), argument_0);
        assert_eq!(op.operand(2).unwrap().value().unwrap(), argument_0);
        assert_eq!(op.operand(0).unwrap().operand_index(), 0);
        assert_eq!(op.operand(1).unwrap().operand_index(), 1);
        assert_eq!(op.operand(2).unwrap().operand_index(), 2);
        assert!(op.operand(3).is_err());
        assert_eq!(op.operand_value(0).unwrap(), argument_0);
        assert_eq!(op.operand_value(1).unwrap(), argument_0);
        assert_eq!(op.operand_value(2).unwrap(), argument_0);
        assert!(op.operand_value(3).is_err());
        assert_eq!(
            op.operand_values().collect::<Result<Vec<_>, _>>().unwrap().into_iter().skip(1).collect::<Vec<_>>(),
            vec![argument_0.clone(), argument_0]
        );
        assert_eq!(op.operand_type(0).unwrap(), index_type);
        assert_eq!(op.operand_type(1).unwrap(), index_type);
        assert_eq!(op.operand_type(2).unwrap(), index_type);
        assert!(op.operand_type(3).is_err());
        assert_eq!(
            op.operand_types().collect::<Result<Vec<_>, _>>().unwrap().into_iter().collect::<Vec<_>>(),
            vec![index_type, index_type, index_type]
        );

        // Try replacing an operand of an operation.
        let i32_type = context.signless_integer_type(32).as_ref();
        let i64_type = context.signless_integer_type(64).as_ref();
        let mut block = context.block(&[(i32_type, location), (i64_type, location)]);
        let mut op = block.append_operation(op).unwrap();
        let argument_1 = block.argument(0).unwrap().as_ref();
        assert!(unsafe { op.replace_operand(0, argument_1) });
        assert_eq!(op.operand(0).unwrap().value().unwrap(), argument_1);
        assert!(unsafe { !op.replace_operand(10, argument_0) });

        // Try replacing all operands of an operation.
        let argument_2 = block.argument(1).unwrap().as_ref();
        assert!(unsafe { op.replace_operands(&[argument_1, argument_2, argument_0]) });
        assert_eq!(op.operand(0).unwrap().value().unwrap(), argument_1);
        assert_eq!(op.operand(1).unwrap().value().unwrap(), argument_2);
        assert!(unsafe { !op.replace_operands(&[argument_2]) });

        // Try replacing all uses of one value inside an operation.
        let mut op = OperationBuilder::new("foo", context.unknown_location())
            .add_operand(argument_0)
            .add_operand(argument_2)
            .add_operand(argument_0)
            .build()
            .unwrap();
        unsafe { op.replace_uses_of_with(argument_0, argument_2) };
        assert_eq!(op.operand(0).unwrap().value().unwrap(), argument_2);
        assert_eq!(op.operand(1).unwrap().value().unwrap(), argument_2);
        assert_eq!(op.operand(2).unwrap().value().unwrap(), argument_2);
    }

    #[test]
    fn test_operation_results() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);

        // Operation with no results.
        let op = OperationBuilder::new("foo", location).build().unwrap();
        assert!(op.result(0).is_err());

        // Operation with two results.
        let op = OperationBuilder::new("test.op", location).add_results(&[i32_type, i64_type]).build().unwrap();
        assert_eq!(op.result_count(), 2);
        assert!(op.result(0).is_ok());
        assert!(op.result(1).is_ok());
        assert!(op.result(2).is_err());
        assert_eq!(op.result_type(0).unwrap(), i32_type);
        assert_eq!(op.result_type(1).unwrap(), i64_type);
        assert!(op.result_type(2).is_err());
        assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().collect::<Vec<_>>().len(), 2);
        assert_eq!(
            op.result_types().collect::<Result<Vec<_>, _>>().unwrap().into_iter().collect::<Vec<_>>(),
            vec![i32_type, i64_type]
        );
    }

    #[test]
    fn test_operation_regions() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();

        // Operation with no regions.
        let op = OperationBuilder::new("foo", location).build().unwrap();
        assert!(op.region(0).is_err());

        // Operation with three regions.
        let region_0 = context.region();
        let region_1 = context.region();
        let region_2 = context.region();
        let op = OperationBuilder::new("foo", location)
            .add_region(region_0)
            .add_region(region_1)
            .add_region(region_2)
            .build()
            .unwrap();
        assert!(!op.is_empty());
        assert_eq!(op.region_count(), 3);
        assert!(op.region(0).is_ok());
        assert!(op.region(1).is_ok());
        assert!(op.region(2).is_ok());
        assert!(op.region(3).is_err());
        assert_eq!(op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().collect::<Vec<_>>().len(), 3);
    }

    #[test]
    fn test_operation_successors() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let block_0 = context.block_with_no_arguments();
        let block_1 = context.block_with_no_arguments();
        let block_2 = context.block_with_no_arguments();
        let op = OperationBuilder::new("test.op", location).add_successors(&[&block_0, &block_1]).build().unwrap();
        assert_eq!(op.successor_count(), 2);
        assert!(op.successor(0).is_ok());
        assert!(op.successor(1).is_ok());
        assert!(op.successor(2).is_err());
        assert_eq!(op.successors().collect::<Result<Vec<_>, _>>().unwrap().into_iter().collect::<Vec<_>>().len(), 2);
        let mut block_3 = context.block_with_no_arguments();
        let mut op = block_3.append_operation(op).unwrap();
        assert!(op.replace_successor(0, &block_2));
        assert!(!op.replace_successor(10, &block_2));
    }

    #[test]
    fn test_operation_parent_block() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let op = OperationBuilder::new("foo", location).build().unwrap();
        assert!(op.parent_block().unwrap().is_none());
        let mut block = context.block_with_no_arguments();
        let op = block.append_operation(op).unwrap();
        assert_eq!(op.parent_block().unwrap(), Some(block.as_ref()));
    }

    #[test]
    fn test_operation_parent_operation() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        let op = OperationBuilder::new("foo", location)
            .add_results(&[context.index_type()])
            .add_region({
                let mut block = context.block_with_no_arguments();
                block.append_operation(OperationBuilder::new("bar", location).build().unwrap()).unwrap();
                block.try_into().unwrap()
            })
            .build()
            .unwrap();
        let op = block.append_operation(op).unwrap();
        assert_eq!(op.parent_operation().unwrap(), None);
        assert_eq!(
            op.region(0)
                .unwrap()
                .blocks()
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .operations()
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .parent_operation()
                .unwrap()
                .unwrap(),
            op
        );
    }

    #[test]
    fn test_operation_is_before_in_block() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        let op_0 = block.append_operation(OperationBuilder::new("op_0", location).build().unwrap()).unwrap();
        let op_1 = block.append_operation(OperationBuilder::new("op_1", location).build().unwrap()).unwrap();
        let op_2 = block.append_operation(OperationBuilder::new("op_2", location).build().unwrap()).unwrap();
        assert!(op_0.is_before_in_block(&op_1));
        assert!(op_0.is_before_in_block(&op_2));
        assert!(op_1.is_before_in_block(&op_2));
        assert!(!op_1.is_before_in_block(&op_0));
        assert!(!op_2.is_before_in_block(&op_0));
        assert!(!op_2.is_before_in_block(&op_1));
    }

    #[test]
    fn test_operation_move_after_and_before() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        let op_0 = block.append_operation(OperationBuilder::new("op_0", location).build().unwrap()).unwrap();
        let op_1 = block.append_operation(OperationBuilder::new("op_1", location).build().unwrap()).unwrap();
        let op_2 = block.append_operation(OperationBuilder::new("op_2", location).build().unwrap()).unwrap();
        unsafe { op_1.move_after(&op_2) };
        assert_eq!(
            block
                .operations()
                .unwrap()
                .map(|op| op.unwrap().name().as_str().unwrap().to_string())
                .collect::<Vec<_>>(),
            vec!["op_0", "op_2", "op_1"],
        );
        unsafe { op_2.move_before(&op_0) };
        assert_eq!(
            block
                .operations()
                .unwrap()
                .map(|op| op.unwrap().name().as_str().unwrap().to_string())
                .collect::<Vec<_>>(),
            vec!["op_2", "op_0", "op_1"],
        );
    }

    #[test]
    fn test_operation_walk_in_pre_order() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        let op = block
            .append_operation(
                OperationBuilder::new("parent", location)
                    .add_results(&[context.index_type()])
                    .add_region({
                        let mut block = context.block_with_no_arguments();
                        block.append_operation(OperationBuilder::new("child_0", location).build().unwrap()).unwrap();
                        block.append_operation(OperationBuilder::new("child_1", location).build().unwrap()).unwrap();
                        block.try_into().unwrap()
                    })
                    .build()
                    .unwrap(),
            )
            .unwrap();

        // Test with `WalkResult::Advance`.
        let mut result: Vec<String> = Vec::new();
        op.walk(WalkOrder::PreOrder, |op| {
            result.push(op.name().as_str().unwrap().to_string());
            WalkResult::Advance
        });
        assert_eq!(vec!["parent", "child_0", "child_1"], result);

        // Test with `WalkResult::Interrupt`.
        result.clear();
        op.walk(WalkOrder::PreOrder, |op| {
            let name = op.name().as_str().unwrap().to_string();
            result.push(name.clone());
            match name.as_str() {
                "parent" => WalkResult::Advance,
                _ => WalkResult::Interrupt,
            }
        });
        assert_eq!(vec!["parent", "child_0"], result);

        // Test with `WalkResult::Skip`.
        result.clear();
        op.walk(WalkOrder::PreOrder, |op| {
            result.push(op.name().as_str().unwrap().to_string());
            WalkResult::Skip
        });
        assert_eq!(vec!["parent"], result);
    }

    #[test]
    fn test_operation_walk_in_post_order() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        let op = block
            .append_operation(
                OperationBuilder::new("grandparent", location)
                    .add_region({
                        let mut block = context.block_with_no_arguments();
                        block
                            .append_operation(
                                OperationBuilder::new("parent", location)
                                    .add_region({
                                        let mut block = context.block_with_no_arguments();
                                        block
                                            .append_operation(OperationBuilder::new("child", location).build().unwrap())
                                            .unwrap();
                                        block.try_into().unwrap()
                                    })
                                    .build()
                                    .unwrap(),
                            )
                            .unwrap();
                        block.try_into().unwrap()
                    })
                    .build()
                    .unwrap(),
            )
            .unwrap();

        // Test with `WalkResult::Advance`.
        let mut result: Vec<String> = Vec::new();
        op.walk(WalkOrder::PostOrder, |op| {
            result.push(op.name().as_str().unwrap().to_string());
            WalkResult::Advance
        });
        assert_eq!(vec!["child", "parent", "grandparent"], result);

        // Test with `WalkResult::Interrupt`.
        result.clear();
        op.walk(WalkOrder::PostOrder, |op| {
            let name = op.name().as_str().unwrap().to_string();
            result.push(name.clone());
            match name.as_str() {
                "child" => WalkResult::Advance,
                _ => WalkResult::Interrupt,
            }
        });
        assert_eq!(vec!["child", "parent"], result);

        // Test with `WalkResult::Skip`, which should result in the same behavior as `WalkResult::Advance`
        // because when walking in `WalkOrder::PostOrder` we always visit children before their parents.
        result.clear();
        op.walk(WalkOrder::PostOrder, |op| {
            result.push(op.name().as_str().unwrap().to_string());
            WalkResult::Skip
        });
        assert_eq!(vec!["child", "parent", "grandparent"], result);
    }

    #[test]
    fn test_operation_bytecode() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
        let op = func::func("test_func", func::FuncAttributes::default(), block.try_into().unwrap(), location).unwrap();
        let bytecode = op.bytecode();
        assert!(bytecode.len() > 0);
        let bytecode = op.bytecode_with_configuration(&BytecodeWriterConfiguration { version: Some(0) });
        assert!(bytecode.is_some());
        assert!(bytecode.unwrap().len() > 0);
        let bytecode = op.bytecode_for_version(0);
        assert!(bytecode.is_some());
        assert!(bytecode.unwrap().len() > 0);
    }

    #[test]
    fn test_operation_to_string_with_flags() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let op = OperationBuilder::new("foo", location).build().unwrap();
        assert_eq!(
            op.to_string_with_flags(OperationPrintingFlags {
                elements_attribute_size_threshold: Some(100),
                enable_debug_information: true,
                use_generic_op_form: true,
                use_local_scope: true,
                ..Default::default()
            })
            .unwrap(),
            "\"foo\"() : () -> () [unknown]"
        );
    }

    #[test]
    fn test_operation_to_string_with_state() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let op = OperationBuilder::new("test.op", location).build().unwrap();
        assert_eq!(
            op.to_string_with_state(AsmState::for_operation(&op, OperationPrintingFlags::default())).unwrap(),
            "\"test.op\"() : () -> ()\n"
        );
    }

    #[test]
    fn test_operation_verify() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();
        let location = context.unknown_location();

        // Valid operation.
        let mut block = context.block_with_no_arguments();
        block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
        let op = func::func("valid", func::FuncAttributes::default(), block.try_into().unwrap(), location).unwrap();
        assert!(op.verify());

        // Invalid operation.
        let op = OperationBuilder::new("unregistered.op", location).build().unwrap();
        assert!(!op.verify());

        // Unregistered but structurally valid operation.
        context.allow_unregistered_dialects();
        let op = OperationBuilder::new("unregistered.op", location).build().unwrap();
        assert!(op.verify());
    }

    #[test]
    fn test_operation_dump() {
        let context = Context::new();
        let op = OperationBuilder::new("foo", context.unknown_location()).build().unwrap();

        // We are just checking that `Operation::dump` runs successfully without crashing.
        // Ideally, we would want a way to capture the standard error stream and verify that it printed the right thing.
        op.dump();
    }

    #[test]
    fn test_operation_clone() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let op_0 = OperationBuilder::new("test.op", location)
            .add_attribute("key", context.string_attribute("value"))
            .build()
            .unwrap();
        let op_1 = op_0.clone();

        // Cloned operations should not be equal.
        assert_ne!(op_0, op_1);

        // Cloned operations should have the same name, attributes, etc.
        assert_eq!(op_0.name(), op_1.name());
        assert_eq!(op_0.attribute("key").unwrap().unwrap(), op_1.attribute("key").unwrap().unwrap());
    }

    #[test]
    fn test_operation_equality_and_hashing() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let op_0 = OperationBuilder::new("test.op", location).build().unwrap();
        let op_1 = OperationBuilder::new("test.op", location).build().unwrap();
        let op_0_ref = op_0.as_ref();
        let op_1_ref = op_1.as_ref();
        assert_eq!(op_0, op_0_ref);
        assert_ne!(op_0, op_1_ref);

        // Test hashing of detached operations.
        let mut map = HashMap::new();
        assert_eq!(map.insert(&op_0, "op_0"), None);
        assert_eq!(map.insert(&op_1, "op_1"), None);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&op_0), Some(&"op_0"));
        assert_eq!(map.get(&op_1), Some(&"op_1"));

        // Test hashing of operation references.
        let mut map = HashMap::new();
        assert_eq!(map.insert(&op_0_ref, "op_0"), None);
        assert_eq!(map.insert(&op_1_ref, "op_1"), None);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&op_0_ref), Some(&"op_0"));
        assert_eq!(map.get(&op_1_ref), Some(&"op_1"));
    }

    #[test]
    fn test_operation_display_and_debug() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let op = OperationBuilder::new("foo", location).build().unwrap();
        assert_eq!(format!("{}", op), "\"foo\"() : () -> ()\n");
        assert_eq!(format!("{:?}", op), "DetachedOperation[\"foo\"() : () -> ()\n]");
        assert_eq!(format!("{}", op.as_ref()), "\"foo\"() : () -> ()\n");
        assert_eq!(format!("{:?}", op.as_ref()), "OperationRef[\"foo\"() : () -> ()\n]");
    }

    #[test]
    fn test_operation_casting() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
        let op = func::func("test_func", func::FuncAttributes::default(), block.try_into().unwrap(), location).unwrap();
        let op = unsafe { op.cast::<DetachedOperation>() };
        assert!(op.is_some());
        let op = op.unwrap();
        let op_ref = unsafe { op.as_ref().cast::<OperationRef>() };
        assert!(op_ref.is_some());
        assert_eq!(op_ref.unwrap().name(), op.name());
    }

    #[test]
    fn test_operation_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();

        // Parse a good operation.
        let op = context.parse_operation("func.func @test() {\n  func.return\n}", "test.mlir");
        assert!(op.is_ok());
        let op = op.unwrap();
        assert!(op.verify());
        assert_eq!(op.name().as_str(), Ok("func.func"));

        // Parse the operation's bytecode directly without interpreting it as UTF-8.
        let bytecode = op.bytecode();
        assert!(bytecode.starts_with(b"ML\xefR"));
        let parsed = context.parse_operation_from_bytes(&bytecode, "test.mlir").unwrap();
        assert!(parsed.verify());
        assert_eq!(parsed.to_string(), op.to_string());
        assert_eq!(parsed.bytecode(), bytecode);

        // Parse invalid bytes and retain the caller-provided filename in both the error and structured diagnostic.
        let diagnostics = Rc::new(RefCell::new(Vec::new()));
        let diagnostics_clone = diagnostics.clone();
        let handler = context.attach_diagnostics_handler(move |diagnostic| {
            diagnostics_clone.borrow_mut().push((
                diagnostic.severity(),
                diagnostic.location().unwrap().to_string(),
                diagnostic.to_string(),
            ));
            true
        });
        assert!(matches!(
            context.parse_operation_from_bytes([0, 255], "invalid-bytes.mlir"),
            Err(Error::ParsingError { message, .. })
                if message == "failed to parse MLIR operation from `invalid-bytes.mlir`",
        ));
        assert_eq!(diagnostics.borrow().len(), 1);
        assert_eq!(diagnostics.borrow()[0].0, DiagnosticSeverity::Error);
        assert!(diagnostics.borrow()[0].1.contains("invalid-bytes.mlir"));
        assert!(!diagnostics.borrow()[0].2.is_empty());
        context.detach_diagnostics_handler(handler);

        // Try parsing a bad operation.
        let op = context.parse_operation("invalid syntax", "invalid.mlir");
        assert!(op.is_err());
    }
}
