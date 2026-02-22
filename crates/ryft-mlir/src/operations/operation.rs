use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use ryft_xla_sys::bindings::{
    MlirOperation, MlirWalkOrder, MlirWalkOrder_MlirWalkPostOrder, MlirWalkOrder_MlirWalkPreOrder, MlirWalkResult,
    MlirWalkResult_MlirWalkResultAdvance, MlirWalkResult_MlirWalkResultInterrupt, MlirWalkResult_MlirWalkResultSkip,
    mlirOperationClone, mlirOperationCreateParse, mlirOperationDestroy, mlirOperationDump, mlirOperationEqual,
    mlirOperationGetAttribute, mlirOperationGetAttributeByName, mlirOperationGetBlock,
    mlirOperationGetDiscardableAttribute, mlirOperationGetDiscardableAttributeByName,
    mlirOperationGetInherentAttributeByName, mlirOperationGetLocation, mlirOperationGetName,
    mlirOperationGetNumAttributes, mlirOperationGetNumDiscardableAttributes, mlirOperationGetNumOperands,
    mlirOperationGetNumRegions, mlirOperationGetNumResults, mlirOperationGetNumSuccessors, mlirOperationGetOperand,
    mlirOperationGetParentOperation, mlirOperationGetRegion, mlirOperationGetResult, mlirOperationGetSuccessor,
    mlirOperationGetTypeID, mlirOperationHasInherentAttributeByName, mlirOperationHashValue,
    mlirOperationIsBeforeInBlock, mlirOperationMoveAfter, mlirOperationMoveBefore, mlirOperationPrint,
    mlirOperationPrintWithFlags, mlirOperationPrintWithState, mlirOperationRemoveAttributeByName,
    mlirOperationRemoveDiscardableAttributeByName, mlirOperationSetAttributeByName,
    mlirOperationSetDiscardableAttributeByName, mlirOperationSetInherentAttributeByName, mlirOperationSetLocation,
    mlirOperationSetOperand, mlirOperationSetOperands, mlirOperationSetSuccessor, mlirOperationVerify,
    mlirOperationWalk, mlirOperationWriteBytecode, mlirOperationWriteBytecodeWithConfig,
};

use crate::support::{write_to_formatter_callback, write_to_string_callback};
use crate::{
    Attribute, AttributeRef, Block, BlockRef, Context, Identifier, Location, LocationRef, LogicalResult,
    NamedAttributeRef, OperationResultRef, RegionRef, StringRef, TypeId, TypeRef, Value, ValueRef,
    write_to_bytes_callback,
};

use super::printing::{AsmState, BytecodeWriterConfiguration, OperationPrintingFlags};

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
    unsafe fn from_c_api(handle: MlirOperation, context: &'c Context<'t>) -> Option<Self>;

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
    fn as_operation_ref(&self) -> OperationRef<'o, 'c, 't> {
        unsafe { OperationRef::from_c_api(self.to_c_api(), self.context()).unwrap() }
    }

    /// Gets the [`TypeId`] of this [`Operation`]. Note that this function may return the same [`TypeId`] for different
    /// instances of the same operation with potentially different attributes. That is because a [`TypeId`] is a unique
    /// identifier of the corresponding MLIR C++ type for the operation and not for a specific instance of this
    /// operation type. Also, note that if the operation does not have a registered description, then this function
    /// will return [`None`].
    fn type_id(&self) -> Option<TypeId<'c>> {
        unsafe { TypeId::from_c_api(mlirOperationGetTypeID(self.to_c_api())) }
    }

    /// Returns the [`Location`] of this [`Operation`].
    fn location(&self) -> LocationRef<'c, 't> {
        unsafe { LocationRef::from_c_api(mlirOperationGetLocation(self.to_c_api()), self.context()).unwrap() }
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
    /// found, then this function returns [`None`].
    ///
    /// Refer to the documentation of [`Operation::attributes`] for information on the distinction between
    /// inherent and discardable attributes.
    fn inherent_attribute<N: AsRef<str>>(&self, name: N) -> Option<AttributeRef<'c, 't>> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe {
            AttributeRef::from_c_api(
                mlirOperationGetInherentAttributeByName(self.to_c_api(), StringRef::from(name.as_ref()).to_c_api()),
                self.context(),
            )
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
        self.discardable_attribute(name).is_some()
    }

    /// Returns a discardable [`Attribute`] of this [`Operation`] with the provided name. If no such attribute can be
    /// found, then this function returns [`None`].
    ///
    /// Refer to the documentation of [`Operation::attributes`] for information on the distinction between
    /// inherent and discardable attributes.
    fn discardable_attribute<N: AsRef<str>>(&self, name: N) -> Option<AttributeRef<'c, 't>> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe {
            AttributeRef::from_c_api(
                mlirOperationGetDiscardableAttributeByName(self.to_c_api(), StringRef::from(name.as_ref()).to_c_api()),
                self.context(),
            )
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
        self.attribute(name).is_some()
    }

    /// Returns an [`Attribute`] of this [`Operation`] with the provided name. If no such attribute can be found,
    /// then this function returns [`None`].
    ///
    /// It is recommended to instead use [`Operation::inherent_attribute`] or
    /// [`Operation::discardable_attribute`]. Refer to the documentation of [`Operation::attributes`] for information
    /// on the distinction between inherent and discardable attributes.
    fn attribute<N: AsRef<str>>(&self, name: N) -> Option<AttributeRef<'c, 't>> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe {
            AttributeRef::from_c_api(
                mlirOperationGetAttributeByName(self.to_c_api(), StringRef::from(name.as_ref()).to_c_api()),
                self.context(),
            )
        }
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
    fn operands<'r>(&'r self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        (0..self.operand_count()).map(|index| unsafe {
            ValueRef::from_c_api(mlirOperationGetOperand(self.to_c_api(), index.cast_signed()), self.context()).unwrap()
        })
    }

    /// Returns the operand at the `index`-pth position in the operands list of this [`Operation`],
    /// and [`None`] if `index` is out of bounds.
    fn operand(&self, index: usize) -> Option<ValueRef<'o, 'c, 't>> {
        if index >= self.operand_count() {
            None
        } else {
            // The following context borrow ensures that access to the underlying MLIR data structures is done safely
            // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
            // to MLIR internals that we have when working with the MLIR C API.
            let _guard = self.context().borrow();
            unsafe {
                ValueRef::from_c_api(mlirOperationGetOperand(self.to_c_api(), index.cast_signed()), self.context())
            }
        }
    }

    /// Returns an [`Iterator`] over the [`Type`](crate::Type)s of the [`Operation::operands`] of this [`Operation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`]
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn operand_types<'r>(&'r self) -> impl Iterator<Item = TypeRef<'c, 't>> {
        self.operands().map(|operand| operand.r#type())
    }

    /// Returns the [`Type`](crate::Type) of the operand at the `index`-pth position in the operands list of this
    /// [`Operation`], and [`None`] if `index` is out of bounds.
    fn operand_type(&self, index: usize) -> Option<TypeRef<'c, 't>> {
        self.operand(index).map(|operand| operand.r#type())
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
    fn results<'r>(&'r self) -> impl Iterator<Item = OperationResultRef<'o, 'c, 't>> {
        (0..self.result_count()).map(|index| unsafe {
            OperationResultRef::from_c_api(mlirOperationGetResult(self.to_c_api(), index.cast_signed()), self.context())
                .unwrap()
        })
    }

    /// Returns the result at the `index`-pth position in the results list of this [`Operation`],
    /// and [`None`] if `index` is out of bounds.
    fn result(&self, index: usize) -> Option<OperationResultRef<'o, 'c, 't>> {
        if index >= self.result_count() {
            None
        } else {
            // The following context borrow ensures that access to the underlying MLIR data structures is done safely
            // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
            // to MLIR internals that we have when working with the MLIR C API.
            let _guard = self.context().borrow();
            unsafe {
                OperationResultRef::from_c_api(
                    mlirOperationGetResult(self.to_c_api(), index.cast_signed()),
                    self.context(),
                )
            }
        }
    }

    /// Returns an [`Iterator`] over the [`Type`](crate::Type)s of the [`Operation::results`] of this [`Operation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`]
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn result_types<'r>(&'r self) -> impl Iterator<Item = TypeRef<'c, 't>> {
        self.results().map(|result| result.r#type())
    }

    /// Returns the [`Type`](crate::Type) of the result at the `index`-pth position in the results list of this
    /// [`Operation`], and [`None`] if `index` is out of bounds.
    fn result_type(&self, index: usize) -> Option<TypeRef<'c, 't>> {
        self.result(index).map(|result| result.r#type())
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
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'o, 'c, 't>> {
        (0..self.region_count()).map(|index| unsafe {
            RegionRef::from_c_api(mlirOperationGetRegion(self.to_c_api(), index.cast_signed()), self.context()).unwrap()
        })
    }

    /// Returns the [`Region`](crate::Region) at the `index`-pth position in the [`Region`](crate::Region)s list
    /// of this [`Operation`], and [`None`] if `index` is out of bounds.
    fn region(&self, index: usize) -> Option<RegionRef<'o, 'c, 't>> {
        if index >= self.region_count() {
            None
        } else {
            // The following context borrow ensures that access to the underlying MLIR data structures is done safely
            // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
            // to MLIR internals that we have when working with the MLIR C API.
            let _guard = self.context().borrow();
            unsafe {
                RegionRef::from_c_api(mlirOperationGetRegion(self.to_c_api(), index.cast_signed()), self.context())
            }
        }
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
    fn successors<'r>(&'r self) -> impl Iterator<Item = BlockRef<'o, 'c, 't>> {
        (0..self.successor_count()).map(|index| unsafe {
            BlockRef::from_c_api(mlirOperationGetSuccessor(self.to_c_api(), index.cast_signed()), self.context())
                .unwrap()
        })
    }

    /// Returns the successor [`Block`] at the `index`-pth position in the successors list of this [`Operation`],
    /// and [`None`] if `index` is out of bounds. Refer to [`Block::successors`] for information
    /// on how successors are defined.
    fn successor(&self, index: usize) -> Option<BlockRef<'o, 'c, 't>> {
        if index >= self.successor_count() {
            None
        } else {
            // The following context borrow ensures that access to the underlying MLIR data structures is done safely
            // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
            // to MLIR internals that we have when working with the MLIR C API.
            let _guard = self.context().borrow();
            unsafe {
                BlockRef::from_c_api(mlirOperationGetSuccessor(self.to_c_api(), index.cast_signed()), self.context())
            }
        }
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
    fn parent_block(&self) -> Option<BlockRef<'o, 'c, 't>> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { BlockRef::from_c_api(mlirOperationGetBlock(self.to_c_api()), self.context()) }
    }

    /// Returns a reference to the parent [`Operation`] of this [`Operation`] (i.e., the [`Operation`] that owns this
    /// operation), if one exists (i.e., if this is not a [`DetachedOperation`] or a reference to a detached
    /// operation).
    fn parent_operation(&self) -> Option<OperationRef<'o, 'c, 't>> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { OperationRef::from_c_api(mlirOperationGetParentOperation(self.to_c_api()), self.context()) }
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
            // We forget `self` and return a new [`OperationRef`] to make sure that ownership is transferred
            // correctly if `self` is a [`DetachedOperation`].
            let context = self.context();
            let handle = self.to_c_api();
            mlirOperationMoveAfter(handle, other.to_c_api());
            if !handle.ptr.is_null() {
                std::mem::forget(self);
            }
            OperationRef::from_c_api(handle, context).unwrap()
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
            // We forget `self` and return a new [`OperationRef`] to make sure that ownership is transferred
            // correctly if `self` is a [`DetachedOperation`].
            let context = self.context();
            let handle = self.to_c_api();
            mlirOperationMoveBefore(handle, other.to_c_api());
            if !handle.ptr.is_null() {
                std::mem::forget(self);
            }
            OperationRef::from_c_api(handle, context).unwrap()
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
                let (ref mut callback, ref context) = *data;
                let operation = OperationRef::from_c_api(operation, context).unwrap();
                (callback)(operation).to_c_api()
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
        let operation = unsafe { O::from_c_api(self.to_c_api(), self.context()) };
        if operation.is_some() {
            std::mem::forget(self);
        }
        operation
    }
}

/// Trait used to represent non-owning references to [`Operation`]s.
pub trait OpRef<'o, 'c: 'o, 't: 'c>: Copy + Clone + Operation<'o, 'c, 't> {
    /// Tries to cast this [`OpRef`] to an instance of `O` (e.g., an instance of [`OperationRef`]). If this
    /// is not an instance of the specified [`OpRef`] type, this function will return [`None`].
    unsafe fn cast<O: OpRef<'o, 'c, 't>>(&self) -> Option<O> {
        unsafe { O::from_c_api(self.to_c_api(), self.context()) }
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
    unsafe fn from_c_api(handle: MlirOperation, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context }) }
    }

    unsafe fn to_c_api(&self) -> MlirOperation {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
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
        write!(formatter, "DetachedOperation[{}]", self.to_string())
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
    /// Parses a [`DetachedOperation`] from the provided string representation. Returns [`None`] if MLIR fails to parse
    /// the provided string into an [`Operation`] (this function will also emit diagnostics if that happens). The
    /// provided `filename` is used to create a [`FileLocationRef`](crate::FileLocationRef) that will be used as the
    /// location of the resulting [`Operation`].
    pub fn parse_operation<'o, 'c: 'o>(&'c self, source: &str, filename: &str) -> Option<DetachedOperation<'c, 't>> {
        unsafe {
            DetachedOperation::from_c_api(
                mlirOperationCreateParse(
                    // The following context borrow ensures that access to the underlying MLIR data structures is done
                    // safely from Rust. It is maybe more conservative than would be ideal, but that is due to the
                    // limited exposure to MLIR internals that we have when working with the MLIR C API.
                    *self.handle.borrow(),
                    StringRef::from(source).to_c_api(),
                    StringRef::from(filename).to_c_api(),
                ),
                &self,
            )
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
    unsafe fn from_c_api(handle: MlirOperation, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context, owner: PhantomData }) }
    }

    unsafe fn to_c_api(&self) -> MlirOperation {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
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
        write!(formatter, "OperationRef[{}]", self.to_string())
    }
}

impl<'r, 'o: 'r, 'c: 'o, 't: 'c, O: DetachedOp<'r, 'c, 't>> From<&'r O> for OperationRef<'o, 'c, 't> {
    fn from(value: &'r O) -> Self {
        unsafe { Self::from_c_api(value.to_c_api(), value.context()).unwrap() }
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
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{
        Block, Context, DetachedModuleOperation, DialectHandle, OperationBuilder, Region, Type, Value, ValueRef,
    };

    use super::*;

    #[test]
    fn test_operation_construction() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());
        context.allow_unregistered_dialects();
        let location = context.unknown_location();

        // Test using a simple unregistered operation that has no type ID.
        let mut op = OperationBuilder::new("foo", location).build().unwrap();
        assert_eq!(op, OperationRef::from(&op));
        assert!(op.type_id().is_none());
        assert_eq!(op.location(), location);
        op.set_location(context.file_location("test.mlir", 4, 2));
        assert_eq!(op.location(), context.file_location("test.mlir", 4, 2));
        assert_eq!(op.name(), context.identifier("foo"));

        // Check that registered operations have type IDs.
        let mut block = context.block_with_no_arguments();
        block.append_operation(func::r#return::<ValueRef, _>(&[], location));
        let op = func::func("test_func", func::FuncAttributes::default(), block.into(), location);
        assert!(op.type_id().is_some());

        // Test a C API-related edge case.
        let op = DetachedOperation { handle: MlirOperation { ptr: std::ptr::null_mut() }, context: &context };
        assert!(unsafe { op.cast::<DetachedModuleOperation>() }.is_none());
    }

    #[test]
    fn test_operation_inherent_attributes() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        block.append_operation(func::r#return::<ValueRef, _>(&[], location));
        let mut op = func::func("test_func", func::FuncAttributes::default(), block.into(), location);
        assert!(op.inherent_attribute_count() > 0);
        assert!(op.has_inherent_attribute("sym_name"));
        assert_eq!(op.inherent_attribute("sym_name"), Some(context.string_attribute("test_func").as_attribute_ref()));
        op.set_inherent_attribute("sym_name", context.string_attribute("modified"));
        assert_eq!(op.inherent_attribute("sym_name"), Some(context.string_attribute("modified").as_attribute_ref()));
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
        assert_eq!(op.discardable_attribute("custom"), Some(context.string_attribute("value").as_attribute_ref()));
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
        assert!(op.attribute("foo").is_some());
        assert_eq!(op.attribute("foo").map(|a| a.to_string()), Some("\"bar\"".into()));
        assert!(op.remove_attribute("foo"));
        assert!(!op.remove_attribute("foo"));
        op.set_attribute("foo", context.string_attribute("foo"));
        assert_eq!(op.attribute("foo").map(|a| a.to_string()), Some("\"foo\"".into()));
        let attribute = op.attributes().next().unwrap();
        assert_eq!(attribute.name(), context.identifier("foo"));
        assert_eq!(attribute.attribute(), context.string_attribute("foo"));
    }

    #[test]
    fn test_operation_operands() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let index_type = context.index_type().as_type_ref();

        // Operation with no operands.
        let op = OperationBuilder::new("foo", location).build().unwrap();
        assert_eq!(op.operands().count(), 0);

        // Operation with three operands.
        let block = context.block(&[(index_type, location)]);
        let argument_0 = block.argument(0).unwrap().as_value_ref();
        let op = OperationBuilder::new("foo", context.unknown_location())
            .add_operand(argument_0)
            .add_operand(argument_0)
            .add_operand(argument_0)
            .build()
            .unwrap();
        assert_eq!(op.operand(0), Some(argument_0));
        assert_eq!(op.operand(1), Some(argument_0));
        assert_eq!(op.operand(2), Some(argument_0));
        assert!(op.operand(3).is_none());
        assert_eq!(op.operands().skip(1).collect::<Vec<_>>(), vec![argument_0.clone(), argument_0]);
        assert_eq!(op.operand_type(0), Some(index_type));
        assert_eq!(op.operand_type(1), Some(index_type));
        assert_eq!(op.operand_type(2), Some(index_type));
        assert!(op.operand_type(3).is_none());
        assert_eq!(op.operand_types().collect::<Vec<_>>(), vec![index_type, index_type, index_type]);

        // Try replacing an operand of an operation.
        let i32_type = context.signless_integer_type(32).as_type_ref();
        let i64_type = context.signless_integer_type(64).as_type_ref();
        let mut block = context.block(&[(i32_type, location), (i64_type, location)]);
        let mut op = block.append_operation(op);
        let argument_1 = block.argument(0).unwrap().as_value_ref();
        assert!(unsafe { op.replace_operand(0, argument_1) });
        assert_eq!(op.operand(0).unwrap(), argument_1);
        assert!(unsafe { !op.replace_operand(10, argument_0) });

        // Try replacing all operands of an operation.
        let argument_2 = block.argument(1).unwrap().as_value_ref();
        assert!(unsafe { op.replace_operands(&[argument_1, argument_2, argument_0]) });
        assert_eq!(op.operand(0), Some(argument_1));
        assert_eq!(op.operand(1), Some(argument_2));
        assert!(unsafe { !op.replace_operands(&[argument_2]) });
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
        assert_eq!(op.result(0), None);

        // Operation with two results.
        let op = OperationBuilder::new("test.op", location).add_results(&[i32_type, i64_type]).build().unwrap();
        assert_eq!(op.result_count(), 2);
        assert!(op.result(0).is_some());
        assert!(op.result(1).is_some());
        assert!(op.result(2).is_none());
        assert_eq!(op.result_type(0).unwrap(), i32_type);
        assert_eq!(op.result_type(1).unwrap(), i64_type);
        assert!(op.result_type(2).is_none());
        assert_eq!(op.results().collect::<Vec<_>>().len(), 2);
        assert_eq!(op.result_types().collect::<Vec<_>>(), vec![i32_type, i64_type]);
    }

    #[test]
    fn test_operation_regions() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();

        // Operation with no regions.
        let op = OperationBuilder::new("foo", location).build().unwrap();
        assert_eq!(op.region(0), None);

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
        assert!(op.region(0).is_some());
        assert!(op.region(1).is_some());
        assert!(op.region(2).is_some());
        assert!(op.region(3).is_none());
        assert_eq!(op.regions().collect::<Vec<_>>().len(), 3);
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
        assert!(op.successor(0).is_some());
        assert!(op.successor(1).is_some());
        assert!(op.successor(2).is_none());
        assert_eq!(op.successors().collect::<Vec<_>>().len(), 2);
        let mut block_3 = context.block_with_no_arguments();
        let mut op = block_3.append_operation(op);
        assert!(op.replace_successor(0, &block_2));
        assert!(!op.replace_successor(10, &block_2));
    }

    #[test]
    fn test_operation_parent_block() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let op = OperationBuilder::new("foo", location).build().unwrap();
        assert!(op.parent_block().is_none());
        let mut block = context.block_with_no_arguments();
        let op = block.append_operation(op);
        assert_eq!(op.parent_block(), Some(block.as_block_ref()));
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
                block.append_operation(OperationBuilder::new("bar", location).build().unwrap());
                block.into()
            })
            .build()
            .unwrap();
        let op = block.append_operation(op);
        assert_eq!(op.parent_operation(), None);
        assert_eq!(
            &op.region(0)
                .unwrap()
                .blocks()
                .next()
                .unwrap()
                .operations()
                .next()
                .unwrap()
                .parent_operation()
                .unwrap(),
            &op
        );
    }

    #[test]
    fn test_operation_is_before_in_block() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        let op_0 = block.append_operation(OperationBuilder::new("op_0", location).build().unwrap());
        let op_1 = block.append_operation(OperationBuilder::new("op_1", location).build().unwrap());
        let op_2 = block.append_operation(OperationBuilder::new("op_2", location).build().unwrap());
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
        let op_0 = block.append_operation(OperationBuilder::new("op_0", location).build().unwrap());
        let op_1 = block.append_operation(OperationBuilder::new("op_1", location).build().unwrap());
        let op_2 = block.append_operation(OperationBuilder::new("op_2", location).build().unwrap());
        unsafe { op_1.move_after(&op_2) };
        assert_eq!(
            block.operations().map(|op| op.name().as_str().unwrap().to_string()).collect::<Vec<_>>(),
            vec!["op_0", "op_2", "op_1"],
        );
        unsafe { op_2.move_before(&op_0) };
        assert_eq!(
            block.operations().map(|op| op.name().as_str().unwrap().to_string()).collect::<Vec<_>>(),
            vec!["op_2", "op_0", "op_1"],
        );
    }

    #[test]
    fn test_operation_walk_in_pre_order() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        let op = block.append_operation(
            OperationBuilder::new("parent", location)
                .add_results(&[context.index_type()])
                .add_region({
                    let mut block = context.block_with_no_arguments();
                    block.append_operation(OperationBuilder::new("child_0", location).build().unwrap());
                    block.append_operation(OperationBuilder::new("child_1", location).build().unwrap());
                    block.into()
                })
                .build()
                .unwrap(),
        );

        // Test with [`WalkResult::Advance`].
        let mut result: Vec<String> = Vec::new();
        op.walk(WalkOrder::PreOrder, |op| {
            result.push(op.name().as_str().unwrap().to_string());
            WalkResult::Advance
        });
        assert_eq!(vec!["parent", "child_0", "child_1"], result);

        // Test with [`WalkResult::Interrupt`].
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

        // Test with [`WalkResult::Skip`].
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
        let op = block.append_operation(
            OperationBuilder::new("grandparent", location)
                .add_region({
                    let mut block = context.block_with_no_arguments();
                    block.append_operation(
                        OperationBuilder::new("parent", location)
                            .add_region({
                                let mut block = context.block_with_no_arguments();
                                block.append_operation(OperationBuilder::new("child", location).build().unwrap());
                                block.into()
                            })
                            .build()
                            .unwrap(),
                    );
                    block.into()
                })
                .build()
                .unwrap(),
        );

        // Test with [`WalkResult::Advance`].
        let mut result: Vec<String> = Vec::new();
        op.walk(WalkOrder::PostOrder, |op| {
            result.push(op.name().as_str().unwrap().to_string());
            WalkResult::Advance
        });
        assert_eq!(vec!["child", "parent", "grandparent"], result);

        // Test with [`WalkResult::Interrupt`].
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

        // Test with [`WalkResult::Skip`], which should result in the same behavior as [`WalkResult::Advance`]
        // because when walking in [`WalkOrder::PostOrder`] we always visit children before their parents.
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
        context.load_dialect(DialectHandle::func());
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        block.append_operation(func::r#return::<ValueRef, _>(&[], location));
        let op = func::func("test_func", func::FuncAttributes::default(), block.into(), location);
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
            }),
            Ok("\"foo\"() : () -> () [unknown]".to_string())
        );
    }

    #[test]
    fn test_operation_to_string_with_state() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let op = OperationBuilder::new("test.op", location).build().unwrap();
        assert_eq!(
            op.to_string_with_state(AsmState::for_operation(&op, OperationPrintingFlags::default())),
            Ok("\"test.op\"() : () -> ()\n".to_string())
        );
    }

    #[test]
    fn test_operation_verify() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());
        let location = context.unknown_location();

        // Valid operation.
        let mut block = context.block_with_no_arguments();
        block.append_operation(func::r#return::<ValueRef, _>(&[], location));
        let op = func::func("valid", func::FuncAttributes::default(), block.into(), location);
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

        // We are just checking that [`Operation::dump`] runs successfully without crashing.
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
        assert_eq!(op_0.attribute("key"), op_1.attribute("key"));
    }

    #[test]
    fn test_operation_equality_and_hashing() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let op_0 = OperationBuilder::new("test.op", location).build().unwrap();
        let op_1 = OperationBuilder::new("test.op", location).build().unwrap();
        let op_0_ref = op_0.as_operation_ref();
        let op_1_ref = op_1.as_operation_ref();
        assert_eq!(op_0, op_0_ref);
        assert_ne!(op_0, op_1_ref);

        // Test hashing of detached operations.
        let mut map = HashMap::new();
        map.insert(&op_0, "op_0");
        map.insert(&op_1, "op_1");
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&op_0), Some(&"op_0"));
        assert_eq!(map.get(&op_1), Some(&"op_1"));

        // Test hashing of operation references.
        let mut map = HashMap::new();
        map.insert(&op_0_ref, "op_0");
        map.insert(&op_1_ref, "op_1");
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
        assert_eq!(format!("{}", op.as_operation_ref()), "\"foo\"() : () -> ()\n");
        assert_eq!(format!("{:?}", op.as_operation_ref()), "OperationRef[\"foo\"() : () -> ()\n]");
    }

    #[test]
    fn test_operation_casting() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        block.append_operation(func::r#return::<ValueRef, _>(&[], location));
        let op = func::func("test_func", func::FuncAttributes::default(), block.into(), location);
        let op = unsafe { op.cast::<DetachedOperation>() };
        assert!(op.is_some());
        let op = op.unwrap();
        let op_ref = unsafe { op.as_operation_ref().cast::<OperationRef>() };
        assert!(op_ref.is_some());
        assert_eq!(op_ref.unwrap().name(), op.name());
    }

    #[test]
    fn test_operation_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());

        // Parse a good operation.
        let op = context.parse_operation("func.func @test() {\n  func.return\n}", "test.mlir");
        assert!(op.is_some());
        let op = op.unwrap();
        assert!(op.verify());
        assert_eq!(op.name().as_str(), Ok("func.func"));

        // Trying parsing a bad operation.
        let op = context.parse_operation("invalid syntax", "invalid.mlir");
        assert!(op.is_none());
    }
}
