use std::cell::Ref;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use ryft_xla_sys::bindings::{
    MlirBlock, MlirContext, MlirOperation, mlirBlockAddArgument, mlirBlockAppendOwnedOperation, mlirBlockCreate,
    mlirBlockDestroy, mlirBlockDetach, mlirBlockEqual, mlirBlockEraseArgument, mlirBlockGetArgument,
    mlirBlockGetFirstOperation, mlirBlockGetNumArguments, mlirBlockGetNumPredecessors, mlirBlockGetNumSuccessors,
    mlirBlockGetParentOperation, mlirBlockGetParentRegion, mlirBlockGetPredecessor, mlirBlockGetSuccessor,
    mlirBlockGetTerminator, mlirBlockInsertArgument, mlirBlockInsertOwnedOperation, mlirBlockInsertOwnedOperationAfter,
    mlirBlockInsertOwnedOperationBefore, mlirBlockPrint, mlirOperationGetNextInBlock, mlirOperationRemoveFromParent,
};

use crate::support::write_to_formatter_callback;
use crate::{
    BlockArgumentRef, Context, DetachedOp, DetachedOperation, Location, LocationRef, OpRef, Operation, OperationRef,
    RegionRef, Type, TypeRef, Value,
};

/// [`Block`]s are one of the main building blocks of MLIR programs. MLIR is fundamentally based on a graph-like
/// data structure of nodes, called [`Operation`]s, and edges, called [`Value`]s. Each [`Value`] is either a
/// [`BlockArgumentRef`] or an [`OperationResultRef`](crate::OperationResultRef), and has a [`Type`] defined by the type
/// system. [`Operation`]s are contained in [`Block`]s and [`Block`]s are contained in [`Region`](crate::Region)s.
/// [`Operation`]s are also ordered within their containing [`Block`] and [`Block`]s are ordered in their containing
/// [`Region`](crate::Region)s, although this order may or may not be semantically meaningful in a given kind of
/// region). [`Operation`]s may also contain [`Region`](crate::Region)s, enabling hierarchical structures to be
/// represented.
///
/// [`Block`]s represent sequences of operations with a single entry point and explicit control flow between them.
/// They're the basic units of control flow, similar to basic blocks in traditional compiler IRs. Each block has
/// predecessors, successors, and block arguments (similar to phi nodes).
///
/// Note that there are multiple separate lifetime parameters: one for the lifetime of the underlying [`Block`], `'b`,
/// one for the [`Context`] which is associated with it, `'c`, and one for the lifetime of the thread pool used by that
/// [`Context`], `'t`. That is because, [`Block`]s can be either owned (i.e., [`DetachedBlock`]s) or borrowed references
/// to underlying MLIR blocks owned by [`Region`](crate::Region)s (i.e., [`BlockRef`]s), which themselves may be owned
/// by [`Operation`]s, etc.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/LangRef/#high-level-structure)
/// for more information.
pub trait Block<'b, 'c: 'b, 't: 'c>: Sized {
    /// Returns the [`MlirBlock`] that corresponds to this [`Block`] and which can be passed to functions
    /// in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn to_c_api(&self) -> MlirBlock;

    /// Returns a reference to the [`Context`] that this [`Block`] is associated with.
    fn context(&self) -> &'c Context<'t>;

    /// Returns a reference to this [`Block`].
    fn as_block_ref(&self) -> BlockRef<'b, 'c, 't> {
        unsafe { BlockRef::from_c_api(self.to_c_api(), self.context()).unwrap() }
    }

    /// Returns the number of [`BlockArgumentRef`]s of this [`Block`].
    fn argument_count(&self) -> usize {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirBlockGetNumArguments(self.to_c_api()).cast_unsigned() }
    }

    /// Returns all [`BlockArgumentRef`]s of this [`Block`].
    fn arguments<'r>(&'r self) -> BlockArgumentRefIterator<'r, 'b, 'c, 't, Self> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        BlockArgumentRefIterator {
            block: &self,
            current_argument_index: 0,
            argument_count: self.argument_count(),
            _context: self.context().borrow(),
            _block: PhantomData,
            _thread_pool: PhantomData,
        }
    }

    /// Returns the [`BlockArgumentRef`] at the `index`-pth position in the arguments list of this [`Block`],
    /// and [`None`] if `index` is out of bounds.
    fn argument(&self, index: usize) -> Option<BlockArgumentRef<'b, 'c, 't>> {
        if index >= self.argument_count() {
            None
        } else {
            unsafe {
                // The following context borrow ensures that access to the underlying MLIR data structures is done
                // safely from Rust. It is maybe more conservative than would be ideal, but that is due to the limited
                // exposure to MLIR internals that we have when working with the MLIR C API.
                let _guard = self.context().borrow();
                BlockArgumentRef::from_c_api(mlirBlockGetArgument(self.to_c_api(), index.cast_signed()), self.context())
            }
        }
    }

    /// Returns `true` if this [`Block`] is empty (i.e., if it contains no [`Operation`]s).
    fn is_empty(&self) -> bool {
        self.operations().current_operation.is_none()
    }

    /// Returns a [`BlockOperationRefIterator`], that enables iteration over references of all [`Operation`]s
    /// contained in this [`Block`].
    fn operations(&self) -> BlockOperationRefIterator<'b, 'c, 't> {
        BlockOperationRefIterator {
            current_operation: unsafe {
                OperationRef::from_c_api(mlirBlockGetFirstOperation(self.to_c_api()), self.context())
            },
        }
    }

    /// Returns a reference to the terminator [`Operation`] of this [`Block`], if one exists.
    ///
    /// In the context of MLIR, an "SSA region terminator" refers to an operation within a [`Region`](crate::Region)
    /// that signals the end of a computational sequence and determines the flow of control. Specifically, in the
    /// context of the `affine` dialect, these terminators often involve [`Operation`]s like `scf.yield` within loop
    /// constructs, or similar operations that define the end of a region and manage the flow of data (SSA values)
    /// within that region. Terminators are crucial for the defining control flow within a region and between
    /// different regions. They determine where execution goes next after the operations within a region are completed.
    /// Examples of terminator operations on MLIR include `cf.br` for branching, `cf.cond_br` for conditional branching,
    /// and `scf.yield` for returning values from loop iterations.
    fn terminator(&self) -> Option<OperationRef<'b, 'c, 't>> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { OperationRef::from_c_api(mlirBlockGetTerminator(self.to_c_api()), self.context()) }
    }

    /// Returns the number of predecessor [`Block`]s of this [`Block`].
    /// Refer to [`Block::predecessors`] for information on how the predecessors of a block are defined.
    fn predecessor_count(&self) -> usize {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirBlockGetNumPredecessors(self.to_c_api()).cast_unsigned() }
    }

    /// Returns references to all predecessor [`Block`]s of this [`Block`]. A block's predecessors are all the blocks
    /// that can directly transfer control to that [`Block`]. In other words:
    ///
    ///   - Block `A` is a predecessor of block `B` if block `B` is a successor of block `A`, following the definition
    ///     of successors provided in [`Block::successors`].
    ///   - Predecessors are the "inverse" relationship of successors.
    ///   - A block can have multiple predecessors if multiple other blocks can branch to it.
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`] because that
    /// would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn predecessors<'r>(&'r self) -> impl Iterator<Item = BlockRef<'b, 'c, 't>> {
        (0..self.predecessor_count()).map(|index| unsafe {
            BlockRef::from_c_api(mlirBlockGetPredecessor(self.to_c_api(), index.cast_signed()), self.context()).unwrap()
        })
    }

    /// Returns a reference to the `index`-th predecessor of this [`Block`], and [`None`] if `index` is out of bounds.
    /// Refer to [`Block::predecessors`] for information on how the successors of a block are defined.
    fn predecessor(&self, index: usize) -> Option<BlockRef<'b, 'c, 't>> {
        if index >= self.predecessor_count() {
            None
        } else {
            // The following context borrow ensures that access to the underlying MLIR data structures is done safely
            // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to
            // MLIR internals that we have when working with the MLIR C API.
            let _guard = self.context().borrow();
            unsafe {
                BlockRef::from_c_api(mlirBlockGetPredecessor(self.to_c_api(), index.cast_signed()), self.context())
            }
        }
    }

    /// Returns the number of successor [`Block`]s of this [`Block`].
    /// Refer to [`Block::successors`] for information on how the successors of a block are defined.
    fn successor_count(&self) -> usize {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirBlockGetNumSuccessors(self.to_c_api()).cast_unsigned() }
    }

    /// Returns references to all successor [`Block`]s of this [`Block`]. A block's successors are the blocks that can
    /// be reached directly from that block through its [`Block::terminator`] operation. Specifically, the
    /// successors are determined by the terminator operation at the end of the block. Each successor corresponds
    /// to a potential control flow target. For example:
    ///
    ///   - If a block ends with a conditional branch (`cf.cond_br`), it will have two successors: one for the
    ///     `true` branch and one for the `false` branch.
    ///   - If a block ends with an unconditional branch (`cf.br`), it has one successor.
    ///   - If a block ends with a return operation (`cf.return`), it typically has no successors.
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`] because that
    /// would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn successors<'r>(&'r self) -> impl Iterator<Item = BlockRef<'b, 'c, 't>> {
        (0..self.successor_count()).map(|index| unsafe {
            BlockRef::from_c_api(mlirBlockGetSuccessor(self.to_c_api(), index.cast_signed()), self.context()).unwrap()
        })
    }

    /// Returns a reference to the `index`-th successor of this [`Block`], and [`None`] if `index` is out of bounds.
    /// Refer to [`Block::successors`] for information on how the successors of a block are defined.
    fn successor(&self, index: usize) -> Option<BlockRef<'b, 'c, 't>> {
        if index >= self.successor_count() {
            None
        } else {
            // The following context borrow ensures that access to the underlying MLIR data structures is done safely
            // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to
            // MLIR internals that we have when working with the MLIR C API.
            let _guard = self.context().borrow();
            unsafe { BlockRef::from_c_api(mlirBlockGetSuccessor(self.to_c_api(), index.cast_signed()), self.context()) }
        }
    }

    /// Appends a [`BlockArgumentRef`] with the specified [`Type`] and [`Location`] to the end of the arguments list
    /// of this [`Block`] and returns it.
    fn append_argument<T: Type<'c, 't>, L: Location<'c, 't>>(
        &self,
        argument_type: T,
        argument_location: L,
    ) -> BlockArgumentRef<'b, 'c, 't> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            BlockArgumentRef::from_c_api(
                mlirBlockAddArgument(self.to_c_api(), argument_type.to_c_api(), argument_location.to_c_api()),
                self.context(),
            )
            .unwrap()
        }
    }

    /// Inserts a [`BlockArgumentRef`] with the specified [`Type`] and [`Location`] at the `index` position in the
    /// arguments list of this [`Block`] and returns it.
    fn insert_argument<T: Type<'c, 't>, L: Location<'c, 't>>(
        &self,
        index: usize,
        argument_type: T,
        argument_location: L,
    ) -> BlockArgumentRef<'b, 'c, 't> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            BlockArgumentRef::from_c_api(
                mlirBlockInsertArgument(
                    self.to_c_api(),
                    index.cast_signed(),
                    argument_type.to_c_api(),
                    argument_location.to_c_api(),
                ),
                self.context(),
            )
            .unwrap()
        }
    }

    /// Removes the [`BlockArgumentRef`] at the `index` position in the arguments list of this [`Block`].
    fn remove_argument(&mut self, index: usize) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe { mlirBlockEraseArgument(self.to_c_api(), index as std::ffi::c_uint) }
    }

    /// Appends the provided [`Operation`] to the end of this [`Block`] and returns a reference
    /// to the appended [`Operation`].
    fn append_operation<'o, O: DetachedOp<'o, 'c, 't>>(&mut self, operation: O) -> OperationRef<'b, 'c, 't>
    where
        'c: 'o,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            let handle = operation.to_c_api();
            std::mem::forget(operation);
            mlirBlockAppendOwnedOperation(self.to_c_api(), handle);
            OperationRef::from_c_api(handle, self.context()).unwrap()
        }
    }

    /// Inserts the provided [`Operation`] at the specified index/position of this [`Block`] and returns a reference
    /// to the inserted [`Operation`]. Note that this is an expensive operation that scans the block linearly to find
    /// the insertion point. You should instead use [`Block::insert_operation_after`] and/or
    /// [`Block::insert_operation_before`] when possible.
    fn insert_operation<'o, O: DetachedOp<'o, 'c, 't>>(
        &mut self,
        operation: O,
        index: usize,
    ) -> OperationRef<'b, 'c, 't>
    where
        'c: 'o,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            let handle = operation.to_c_api();
            std::mem::forget(operation);
            mlirBlockInsertOwnedOperation(self.to_c_api(), index.cast_signed(), handle);
            OperationRef::from_c_api(handle, self.context()).unwrap()
        }
    }

    /// Inserts the provided [`Operation`] after the provided `reference` operation in this [`Block`] and returns a
    /// reference to the inserted [`Operation`]. The reference operation must belong to this block. If the reference
    /// operation is [`None`], then this function will prepend the provided [`Block`] to this block.
    fn insert_operation_after<'o, O: DetachedOp<'o, 'c, 't>, R: OpRef<'b, 'c, 't>>(
        &mut self,
        operation: O,
        reference: Option<R>,
    ) -> OperationRef<'b, 'c, 't>
    where
        'c: 'o,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            let reference = reference
                .map(|operation| operation.to_c_api())
                .unwrap_or(MlirOperation { ptr: std::ptr::null_mut() });
            let handle = operation.to_c_api();
            std::mem::forget(operation);
            mlirBlockInsertOwnedOperationAfter(self.to_c_api(), reference, handle);
            OperationRef::from_c_api(handle, self.context()).unwrap()
        }
    }

    /// Inserts the provided [`Operation`] before the provided `reference` operation in this [`Block`] and returns a
    /// reference to the inserted [`Operation`]. The reference operation must belong to this block. If the reference
    /// operation is [`None`], then this function will append the provided [`Block`] to this block.
    fn insert_operation_before<'o, O: DetachedOp<'o, 'c, 't>, R: OpRef<'b, 'c, 't>>(
        &mut self,
        operation: O,
        reference: Option<R>,
    ) -> OperationRef<'b, 'c, 't>
    where
        'c: 'o,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            let reference = reference
                .map(|operation| operation.to_c_api())
                .unwrap_or(MlirOperation { ptr: std::ptr::null_mut() });
            let handle = operation.to_c_api();
            std::mem::forget(operation);
            mlirBlockInsertOwnedOperationBefore(self.to_c_api(), reference, handle);
            OperationRef::from_c_api(handle, self.context()).unwrap()
        }
    }

    /// Removes the provided operation from this [`Block`], taking ownership and returning the (potentially) updated
    /// [`Block`] and the owned [`Operation`]. Note that this function will simply do nothing if the provided operation
    /// does not belong to this [`Block`] and return [`None`].
    ///
    /// This function is marked unsafe because if any of the results of the removed operation are still be referenced
    /// in this [`Block`] after this function is called, then those references can become dangling if the returned
    /// [`Operation`] is dropped and cause memory errors (and unfortunately there is no way to protect against this
    /// from Rust).
    unsafe fn remove_operation<O: OpRef<'b, 'c, 't>>(self, operation: O) -> (Self, Option<DetachedOperation<'c, 't>>) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let context = self.context();
        unsafe {
            let handle = self.to_c_api();
            match operation.parent_block() {
                Some(block) if mlirBlockEqual(handle, block.to_c_api()) => {
                    let _guard = context.borrow_mut();
                    let operation_handle = operation.to_c_api();
                    mlirOperationRemoveFromParent(operation_handle);
                    (self, DetachedOperation::from_c_api(operation_handle, context))
                }
                _ => (self, None),
            }
        }
    }
}

/// [`Block`] that is not part of an MLIR program (i.e., it is "detached") and is not owned by the current [`Context`].
/// [`DetachedBlock`]s can be inserted into [`Region`](crate::Region)s, handing off ownership to the respective region.
/// While it is not strictly necessary that a [`DetachedBlock`] keeps a pointer to an MLIR [`Context`] (and its
/// lifetimes), this structure does keep that pointer around (and its lifetimes) as a means to provide more safety when
/// accessing and potentially mutating objects nested inside [`DetachedBlock`]s. Note that this is technically also more
/// "correct" in that there are objects referenced by even [`DetachedBlock`]s that are owned and managed by the
/// associated [`Context`] (e.g., [`Location`]s and [`Type`]s).
pub struct DetachedBlock<'c, 't> {
    /// Handle that represents this [`Block`] in the MLIR C API.
    handle: MlirBlock,

    /// [`Context`] associated with this [`Block`].
    context: &'c Context<'t>,
}

impl<'b, 'c: 'b, 't: 'c> Block<'b, 'c, 't> for DetachedBlock<'c, 't> {
    unsafe fn to_c_api(&self) -> MlirBlock {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

impl<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>> PartialEq<B> for DetachedBlock<'c, 't> {
    fn eq(&self, other: &B) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        // Note that this function only checks for whether the two block handles point to the same underlying block.
        // It does not perform a deep comparison of the contents of these blocks.
        unsafe { mlirBlockEqual(self.handle, other.to_c_api()) }
    }
}

impl<'c, 't> Eq for DetachedBlock<'c, 't> {}

impl<'c, 't> Display for DetachedBlock<'c, 't> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirBlockPrint(
                self.to_c_api(),
                Some(write_to_formatter_callback),
                &mut data as *mut _ as *mut std::ffi::c_void,
            );
        }
        data.1
    }
}

impl<'c, 't> Debug for DetachedBlock<'c, 't> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Block[{}]", self.to_string())
    }
}

impl Drop for DetachedBlock<'_, '_> {
    fn drop(&mut self) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            mlirBlockDestroy(self.handle);
        }
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`DetachedBlock`] with the provided input argument [`Type`]s and [`Location`]s,
    /// and an empty body, which is associated with this [`Context`].
    pub fn block<'c, T: Type<'c, 't>, L: Location<'c, 't>>(&'c self, arguments: &[(T, L)]) -> DetachedBlock<'c, 't> {
        unsafe {
            let argument_types = arguments.iter().map(|arg| arg.0.to_c_api()).collect::<Vec<_>>();
            let argument_locations = arguments.iter().map(|arg| arg.1.to_c_api()).collect::<Vec<_>>();
            DetachedBlock {
                handle: mlirBlockCreate(
                    argument_types.len().cast_signed(),
                    argument_types.as_ptr() as *const _,
                    argument_locations.as_ptr() as *const _,
                ),
                context: &self,
            }
        }
    }

    /// Creates a new empty [`DetachedBlock`] associated with this [`Context`].
    pub fn block_with_no_arguments<'c>(&'c self) -> DetachedBlock<'c, 't> {
        self.block::<TypeRef<'_, '_>, LocationRef<'_, '_>>(&[])
    }
}

/// Reference to an MLIR [`Block`] that is owned by a [`Region`](crate::Region).
///
/// Note that there are multiple separate lifetime parameters: one for the lifetime of the owner of this [`Block`]
/// reference, `'r`, one for the [`Context`] which is associated with it, `'c`, and one for the lifetime of the thread
/// pool used by that [`Context`], `'t`.
#[derive(Copy, Clone)]
pub struct BlockRef<'r, 'c: 'r, 't: 'c> {
    /// Handle that represents this [`Block`] reference in the MLIR C API.
    handle: MlirBlock,

    /// [`Context`] associated with this [`Block`] reference.
    context: &'c Context<'t>,

    /// [`PhantomData`] used to track the lifetime of the [`Region`](crate::Region) that owns the underlying [`Block`].
    owner: PhantomData<&'r ()>,
}

impl<'r, 'c, 't> BlockRef<'r, 'c, 't> {
    /// Constructs a new [`BlockRef`] from the provided [`MlirBlock`] handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub(crate) unsafe fn from_c_api(handle: MlirBlock, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context, owner: PhantomData }) }
    }

    /// Returns a reference to the parent [`Region`](crate::Region) of the referenced [`Block`] (i.e., the closest
    /// surrounding [`Region`](crate::Region) that contains this block).
    pub fn parent_region(&self) -> RegionRef<'r, 'c, 't> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { RegionRef::from_c_api(mlirBlockGetParentRegion(self.to_c_api()), self.context()).unwrap() }
    }

    /// Returns a reference to the parent [`Operation`] of the referenced [`Block`] (i.e., the closest surrounding
    /// [`Operation`] that contains this block), if one exists.
    pub fn parent_operation(&self) -> Option<OperationRef<'r, 'c, 't>> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { OperationRef::from_c_api(mlirBlockGetParentOperation(self.to_c_api()), self.context()) }
    }

    /// Detaches the referenced [`Block`] from its owning [`Region`](crate::Region) and assumes ownership of it,
    /// returning it as a [`DetachedBlock`].
    pub fn detach(self) -> DetachedBlock<'c, 't> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            let context = self.context();
            let handle = self.to_c_api();
            mlirBlockDetach(handle);
            DetachedBlock { handle, context }
        }
    }
}

impl<'r, 'b: 'r, 'c: 'b, 't: 'c> Block<'b, 'c, 't> for BlockRef<'r, 'c, 't> {
    unsafe fn to_c_api(&self) -> MlirBlock {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

impl<'b, 'r: 'b, 'c, 't, B: Block<'b, 'c, 't>> PartialEq<B> for BlockRef<'r, 'c, 't> {
    fn eq(&self, other: &B) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        // Note that this function only checks for whether the two block handles point to the same underlying block.
        // It does not perform a deep comparison of the contents of these blocks.
        unsafe { mlirBlockEqual(self.handle, other.to_c_api()) }
    }
}

impl<'r, 'c, 't> Eq for BlockRef<'r, 'c, 't> {}

impl<'r, 'c, 't> Display for BlockRef<'r, 'c, 't> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirBlockPrint(
                self.to_c_api(),
                Some(write_to_formatter_callback),
                &mut data as *mut _ as *mut std::ffi::c_void,
            );
        }
        data.1
    }
}

impl<'r, 'c, 't> Debug for BlockRef<'r, 'c, 't> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "BlockRef[{}]", self.to_string())
    }
}

impl<'b, 'c, 't> From<&'b DetachedBlock<'c, 't>> for BlockRef<'b, 'c, 't> {
    fn from(value: &'b DetachedBlock<'c, 't>) -> Self {
        value.as_block_ref()
    }
}

/// [`Iterator`] over references to the arguments of a [`Block`].
pub struct BlockArgumentRefIterator<'r, 'b: 'r, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>> {
    /// [`Block`] over whose arguments this iterator is iterating.
    block: &'r B,

    /// Current [`Block`] argument index for this iterator (i.e., the [`BlockArgumentRef`] that will be returned in the
    /// next call to [`BlockArgumentRefIterator::next`] will be referencing the `current_argument_index`-th argument of
    /// the [`Block`]).
    current_argument_index: usize,

    /// Number of [`Block`] arguments over which this iterator is iterating.
    argument_count: usize,

    /// [`Context`] reference that, while unused, ensures that the owning context is not modified while
    /// iterating over arguments using this iterator.
    _context: Ref<'c, MlirContext>,

    /// [`PhantomData`] used to track the lifetime of the underlying [`Block`].
    _block: PhantomData<&'b ()>,

    /// [`PhantomData`] used to track the lifetime of the underlying [`ThreadPool`].
    _thread_pool: PhantomData<&'t ()>,
}

impl<'r, 'b: 'r, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>> Iterator for BlockArgumentRefIterator<'r, 'b, 'c, 't, B> {
    type Item = BlockArgumentRef<'b, 'c, 't>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_argument_index >= self.argument_count {
            None
        } else {
            let current_argument = unsafe {
                BlockArgumentRef::from_c_api(
                    mlirBlockGetArgument(self.block.to_c_api(), self.current_argument_index.cast_signed()),
                    self.block.context(),
                )
                .unwrap()
            };
            self.current_argument_index += 1;
            Some(current_argument)
        }
    }
}

/// [`Iterator`] over references of the [`Operation`]s contained in a [`Block`].
///
/// Note that this iterator does not hold a borrowed reference to the underlying [`Context`] because that would make
/// it impossible to perform mutating operations on that context (e.g., from within [`Pass`](crate::Pass)es) while
/// iterating over the contents of this iterator.
pub struct BlockOperationRefIterator<'b, 'c: 'b, 't: 'c> {
    /// Current [`OperationRef`] in this iterator (i.e., the [`OperationRef`] that will be returned in the next call to
    /// [`BlockOperationRefIterator::next`]). [`OperationRef`]s are stored in such a way in MLIR that we can always
    /// obtain the next [`Operation`] in a [`Block`] given an [`OperationRef`] in that same [`Block`].
    current_operation: Option<OperationRef<'b, 'c, 't>>,
}

impl<'b, 'c, 't> Iterator for BlockOperationRefIterator<'b, 'c, 't> {
    type Item = OperationRef<'b, 'c, 't>;

    fn next(&mut self) -> Option<Self::Item> {
        let current_operation = self.current_operation.take();
        self.current_operation = current_operation.as_ref().and_then(|operation| unsafe {
            OperationRef::from_c_api(mlirOperationGetNextInBlock(operation.to_c_api()), operation.context())
        });
        current_operation
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, DialectHandle, OperationBuilder, Region, ValueRef};

    use super::*;

    #[test]
    fn test_block_construction() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);

        // Empty block with no arguments.
        let block = context.block_with_no_arguments();
        assert_eq!(block.argument_count(), 0);
        assert!(block.is_empty());

        // Empty block with arguments.
        let block = context.block(&[(i32_type, location), (i64_type, location)]);
        assert_eq!(block.argument_count(), 2);
        assert_eq!(block.argument(0).unwrap().r#type(), i32_type);
        assert_eq!(block.argument(1).unwrap().r#type(), i64_type);
        assert!(block.argument(2).is_none());
        assert_eq!(block.arguments().map(|argument| argument.r#type()).collect::<Vec<_>>(), vec![i32_type, i64_type]);
    }

    #[test]
    fn test_block_operations() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        let op_0 = block.append_operation(OperationBuilder::new("foo", location).build().unwrap());
        let op_1 = block.append_operation(OperationBuilder::new("bar", location).build().unwrap());
        assert_eq!(block.operations().collect::<Vec<_>>(), vec![op_0, op_1]);
    }

    #[test]
    fn test_block_terminator() {
        let context = Context::new();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        assert_eq!(block.terminator(), None);
        let op = block.append_operation(func::r#return::<ValueRef, _>(&[], location));
        assert_eq!(block.terminator(), Some(op));
    }

    #[test]
    fn test_block_successors_and_predecessors() {
        let context = Context::new();
        context.load_dialect(DialectHandle::cf());
        let location = context.unknown_location();

        // Create a region with three blocks that are initially all empty and not connected.
        let mut region = context.region();
        let mut block_0 = region.append_block(context.block_with_no_arguments());
        let block_1 = region.append_block(context.block_with_no_arguments());
        let block_2 = region.append_block(context.block_with_no_arguments());
        assert_eq!(block_0.successor_count(), 0);
        assert_eq!(block_0.predecessor_count(), 0);
        assert_eq!(block_1.predecessor_count(), 0);
        assert_eq!(block_2.predecessor_count(), 0);

        // Create a conditional branch from `block_0` to `block_1` and `block_2`.
        let i1_type = context.signless_integer_type(1);
        let condition = block_0.append_argument(i1_type, location);
        let branch_op = OperationBuilder::new("cf.cond_br", location)
            .add_operands(&[condition])
            .add_successors(&[&block_1, &block_2])
            .build()
            .unwrap();
        block_0.append_operation(branch_op);

        // Now `block_0 should have 2 successors and `block_1` and `block_2` should each have 1 predecessor.
        assert_eq!(block_0.successor_count(), 2);
        assert_eq!(block_0.successor(0), Some(block_1));
        assert_eq!(block_0.successor(1), Some(block_2));
        assert_eq!(block_0.successor(2), None);
        assert_eq!(block_0.successors().collect::<Vec<_>>(), vec![block_1, block_2]);
        assert_eq!(block_0.predecessor_count(), 0);
        assert_eq!(block_1.successor_count(), 0);
        assert_eq!(block_1.predecessor_count(), 1);
        assert_eq!(block_1.predecessor(0), Some(block_0));
        assert_eq!(block_1.predecessor(1), None);
        assert_eq!(block_1.predecessors().collect::<Vec<_>>(), vec![block_0]);
        assert_eq!(block_2.successor_count(), 0);
        assert_eq!(block_2.predecessor_count(), 1);
        assert_eq!(block_2.predecessor(0), Some(block_0));
        assert_eq!(block_2.predecessor(1), None);
        assert_eq!(block_2.predecessors().collect::<Vec<_>>(), vec![block_0]);
    }

    #[test]
    fn test_block_append_argument() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let block = context.block_with_no_arguments();
        assert_eq!(block.argument_count(), 0);
        assert!(block.is_empty());
        let argument = block.append_argument(i32_type, location);
        assert_eq!(block.argument_count(), 1);
        assert_eq!(argument.r#type(), i32_type);
    }

    #[test]
    fn test_block_insert_argument() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let block = context.block(&[(i32_type, location)]);
        let argument = block.insert_argument(0, i64_type, location);
        assert_eq!(argument.r#type(), i64_type);
        assert_eq!(block.argument_count(), 2);
        assert_eq!(block.argument(0).unwrap().r#type(), i64_type);
        assert_eq!(block.argument(1).unwrap().r#type(), i32_type);
        assert!(block.argument(2).is_none());
    }

    #[test]
    fn test_block_remove_argument() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let mut block = context.block(&[(i32_type, location), (i64_type, location)]);
        assert_eq!(block.argument_count(), 2);
        block.remove_argument(0);
        assert_eq!(block.argument_count(), 1);
        assert_eq!(block.argument(0).unwrap().r#type(), i64_type);
        assert!(block.argument(1).is_none());
    }

    #[test]
    fn test_block_append_operation() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        let op = block.append_operation(OperationBuilder::new("foo", location).build().unwrap());
        assert_eq!(block.operations().collect::<Vec<_>>(), vec![op]);
    }

    #[test]
    fn test_block_insert_operation() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        let op_0 = block.append_operation(OperationBuilder::new("foo", location).build().unwrap());
        let op_1 = block.insert_operation(OperationBuilder::new("bar", location).build().unwrap(), 0);
        assert_eq!(block.operations().collect::<Vec<_>>(), vec![op_1, op_0]);
    }

    #[test]
    fn test_block_insert_operation_after() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        let op_0 = block.append_operation(OperationBuilder::new("foo", location).build().unwrap());
        let op_1 = block.insert_operation_after(OperationBuilder::new("bar", location).build().unwrap(), Some(op_0));
        let op_2 =
            block.insert_operation_after(OperationBuilder::new("baz", location).build().unwrap(), None::<OperationRef>);
        assert_eq!(block.operations().collect::<Vec<_>>(), vec![op_2, op_0, op_1]);
    }

    #[test]
    fn test_block_insert_operation_before() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let mut block = context.block_with_no_arguments();
        let op_0 = block.append_operation(OperationBuilder::new("foo", location).build().unwrap());
        let op_1 = block.insert_operation_before(OperationBuilder::new("bar", location).build().unwrap(), Some(op_0));
        let op_2 = block
            .insert_operation_before(OperationBuilder::new("baz", location).build().unwrap(), None::<OperationRef>);
        assert_eq!(block.operations().collect::<Vec<_>>(), vec![op_1, op_0, op_2]);
    }

    #[test]
    fn test_block_remove_operation() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();

        // Create a block with two operations and remove one of them.
        let mut block_0 = context.block_with_no_arguments();
        let op_0 = block_0.append_operation(OperationBuilder::new("foo", location).build().unwrap());
        let op_1 = block_0.append_operation(OperationBuilder::new("bar", location).build().unwrap());
        assert_eq!(block_0.operations().collect::<Vec<_>>(), vec![op_0, op_1]);
        let (block, removed_op) = unsafe { block_0.remove_operation(op_0) };
        assert!(removed_op.is_some());
        assert_eq!(removed_op.unwrap(), op_0);
        assert_eq!(block.operations().collect::<Vec<_>>(), vec![op_1]);

        // Create another block and try to remove an operation that is in that block from the previous one.
        let mut block_1 = context.block_with_no_arguments();
        let op_2 = block_1.append_operation(OperationBuilder::new("foo", location).build().unwrap());
        let (block_0, removed_op) = unsafe { block.remove_operation(op_2) };
        assert!(removed_op.is_none());

        // Try to remove an orphaned operation.
        let op_3 = OperationBuilder::new("foo", location).build().unwrap();
        let (_, removed_op) = unsafe { block_0.remove_operation(op_3.as_operation_ref()) };
        assert!(removed_op.is_none());
    }

    #[test]
    fn test_block_ref_parent_region() {
        let context = Context::new();
        let mut region = context.region();
        let block = context.block_with_no_arguments();
        let block = region.append_block(block);
        assert_eq!(block.parent_region(), region);
    }

    #[test]
    fn test_block_ref_parent_operation() {
        let context = Context::new();
        let module = context.module(context.unknown_location());
        assert_eq!(module.body().parent_operation(), Some(module.as_operation().as_operation_ref()));
    }

    #[test]
    fn test_block_detach() {
        let context = Context::new();
        let mut region = context.region();
        let block = region.append_block(context.block_with_no_arguments());
        let detached = block.detach();
        assert_eq!(detached.to_string(), "<<UNLINKED BLOCK>>\n");
    }

    #[test]
    fn test_block_equality() {
        let context = Context::new();
        let mut region = context.region();
        let block_0 = context.block_with_no_arguments();
        let block_1 = region.append_block(context.block_with_no_arguments());
        let block_2 = BlockRef::from(&block_0);
        assert_eq!(block_0, block_0);
        assert_eq!(block_1, block_1);
        assert_eq!(block_2, block_2);
        assert_ne!(block_0, block_1);
        assert_ne!(block_1, block_2);
        assert_eq!(block_2, block_0);
    }

    #[test]
    fn test_block_display_and_debug() {
        let context = Context::new();
        let block = context.block_with_no_arguments();
        assert_eq!(format!("{}", block), "<<UNLINKED BLOCK>>\n");
        assert_eq!(format!("{:?}", block), "Block[<<UNLINKED BLOCK>>\n]");
        assert_eq!(format!("{}", block.as_block_ref()), "<<UNLINKED BLOCK>>\n");
        assert_eq!(format!("{:?}", block.as_block_ref()), "BlockRef[<<UNLINKED BLOCK>>\n]");

        let mut region = context.region();
        let block = region.append_block(context.block_with_no_arguments());
        assert_eq!(format!("{}", block), "<<UNLINKED BLOCK>>\n");
        assert_eq!(format!("{:?}", block), "BlockRef[<<UNLINKED BLOCK>>\n]");
    }
}
