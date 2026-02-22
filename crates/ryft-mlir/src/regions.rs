use std::marker::PhantomData;

use ryft_xla_sys::bindings::{
    MlirBlock, MlirRegion, mlirBlockGetNextInRegion, mlirRegionAppendOwnedBlock, mlirRegionCreate, mlirRegionDestroy,
    mlirRegionEqual, mlirRegionGetFirstBlock, mlirRegionInsertOwnedBlock, mlirRegionInsertOwnedBlockAfter,
    mlirRegionInsertOwnedBlockBefore, mlirRegionTakeBody,
};

use crate::{Block, BlockRef, Context, DetachedBlock, FromWithContext};

/// [`Region`]s are one of the main building blocks of MLIR programs. MLIR is fundamentally based on a graph-like
/// data structure of nodes, called [`Operation`](crate::Operation)s, and edges, called [`Value`](crate::Value)s.
/// Each [`Value`](crate::Value) is either a [`BlockArgumentRef`](crate::BlockArgumentRef) or an
/// [`OperationResultRef`](crate::OperationResultRef), and has a [`Type`](crate::Type) defined by the type system.
/// [`Operation`](crate::Operation)s are contained in [`Block`]s and [`Block`]s are contained in [`Region`]s.
/// [`Operation`](crate::Operation)s are also ordered within their containing [`Block`] and [`Block`]s are ordered in
/// their containing [`Region`]s, although this order may or may not be semantically meaningful in a given kind of
/// region). [`Operation`](crate::Operation)s may also contain [`Region`]s, enabling hierarchical structures to be
/// represented.
///
/// [`Region`]s represent nested scopes or contexts that can contain multiple [`Block`]s. They provide hierarchical
/// structure and encapsulation; they are kind of equivalent to containers that can hold complex control flow graphs
/// while presenting a clean interface to the outside.
///
/// Note that there are multiple separate lifetime parameters: one for the lifetime of the underlying [`Region`], `'r`,
/// one for the [`Context`] which is associated with it, `'c`, and one for the lifetime of the thread pool used by
/// that [`Context`], `'t`. That is because, [`Region`]s can be either owned (i.e., [`DetachedRegion`]s) or borrowed
/// references to underlying MLIR regions owned by [`Operation`](crate::Operation)s (i.e., [`RegionRef`]s), which
/// themselves may be owned by [`Block`]s, etc.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/LangRef/#high-level-structure)
/// for more information.
pub trait Region<'r, 'c: 'r, 't: 'c> {
    /// Returns the [`MlirRegion`] that corresponds to this [`Region`] and which can be passed to functions
    /// in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn to_c_api(&self) -> MlirRegion;

    /// Returns a reference to the [`Context`] that this [`Region`] is associated with.
    fn context(&self) -> &'c Context<'t>;

    /// Returns a reference to this [`Region`].
    fn as_ref(&self) -> RegionRef<'r, 'c, 't> {
        unsafe { RegionRef::from_c_api(self.to_c_api(), self.context()).unwrap() }
    }

    /// Returns `true` if this [`Region`] is empty (i.e., if it contains no [`Block`]s).
    fn is_empty(&self) -> bool {
        self.blocks().current_block.is_none()
    }

    /// Returns a [`RegionBlockRefIterator`], that enables iteration over references of all [`Block`]s contained
    /// in this [`Region`].
    fn blocks(&self) -> RegionBlockRefIterator<'r, 'c, 't> {
        RegionBlockRefIterator {
            current_block: unsafe { BlockRef::from_c_api(mlirRegionGetFirstBlock(self.to_c_api()), self.context()) },
        }
    }

    /// Appends the provided [`Block`] to the end of this [`Region`] and returns a reference to the appended [`Block`].
    fn append_block(&mut self, block: DetachedBlock<'c, 't>) -> BlockRef<'r, 'c, 't> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            let handle = block.to_c_api();
            std::mem::forget(block);
            mlirRegionAppendOwnedBlock(self.to_c_api(), handle);
            BlockRef::from_c_api(handle, self.context()).unwrap()
        }
    }

    /// Inserts the provided [`Block`] at the specified index/position of this [`Region`] and returns a reference to
    /// the inserted [`Block`]. Note that this is an expensive operation that scans the [`Region`] linearly to find the
    /// insertion point. You should instead use [`Region::insert_block_after`] and/or [`Region::insert_block_before`]
    /// when possible.
    fn insert_block(&mut self, block: DetachedBlock<'c, 't>, index: usize) -> BlockRef<'r, 'c, 't> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            let handle = block.to_c_api();
            std::mem::forget(block);
            mlirRegionInsertOwnedBlock(self.to_c_api(), index.cast_signed(), handle);
            BlockRef::from_c_api(handle, self.context()).unwrap()
        }
    }

    /// Inserts the provided [`Block`] after the provided `reference` [`Block`] in this [`Region`] and returns
    /// a reference to the inserted [`Block`]. The `reference` [`Block`] must belong to this [`Region`]. If the
    /// `reference` [`Block`] is [`None`], then this function will prepend the provided [`Block`] to this region.
    fn insert_block_after(
        &mut self,
        block: DetachedBlock<'c, 't>,
        reference: Option<&BlockRef<'r, 'c, 't>>,
    ) -> BlockRef<'r, 'c, 't> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            let reference = reference.map(|block| block.to_c_api()).unwrap_or(MlirBlock { ptr: std::ptr::null_mut() });
            let handle = block.to_c_api();
            std::mem::forget(block);
            mlirRegionInsertOwnedBlockAfter(self.to_c_api(), reference, handle);
            BlockRef::from_c_api(handle, self.context()).unwrap()
        }
    }

    /// Inserts the provided [`Block`] before the provided `reference` [`Block`] in this [`Region`] and returns
    /// a reference to the inserted [`Block`]. The `reference` [`Block`] must belong to this [`Region`]. If the
    /// `reference` [`Block`] is [`None`], then this function will append the provided [`Block`] to this region.
    fn insert_block_before(
        &mut self,
        block: DetachedBlock<'c, 't>,
        reference: Option<&BlockRef<'r, 'c, 't>>,
    ) -> BlockRef<'r, 'c, 't> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            let reference = reference.map(|block| block.to_c_api()).unwrap_or(MlirBlock { ptr: std::ptr::null_mut() });
            let handle = block.to_c_api();
            std::mem::forget(block);
            mlirRegionInsertOwnedBlockBefore(self.to_c_api(), reference, handle);
            BlockRef::from_c_api(handle, self.context()).unwrap()
        }
    }

    /// Takes the body of the `other` [`Region`] and moves it to become the body of this [`Region`], clearing its
    /// pre-existing body. After this function returns, the `other` [`Region`] will have an empty body.
    fn take_body<'o, R: Region<'o, 'c, 't>>(&self, other: R)
    where
        'c: 'o,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirRegionTakeBody(self.to_c_api(), other.to_c_api()) }
    }
}

/// [`Region`] that is not part of an MLIR program (i.e., it is "detached") and is not owned by the current
/// [`Context`]. [`DetachedRegion`]s can be inserted into [`Block`]s, handing off ownership to the respective
/// [`Block`]. While it is not strictly necessary that a [`DetachedRegion`] keeps a pointer to an MLIR [`Context`]
/// (and its lifetimes), this structure does keep that pointer around (and its lifetimes) as a means to provide more
/// safety when accessing and potentially mutating objects nested inside [`DetachedRegion`]s. Note that this is
/// technically also more "correct" in that there are objects referenced by even [`DetachedRegion`]s that are owned
/// and managed by the associated [`Context`] (e.g., [`Location`](crate::Location)s and [`Type`](crate::Type)s).
#[derive(Debug)]
pub struct DetachedRegion<'c, 't> {
    /// Handle that represents this [`Region`] in the MLIR C API.
    handle: MlirRegion,

    /// [`Context`] associated with this [`Region`].
    context: &'c Context<'t>,
}

impl<'r, 'c: 'r, 't: 'c> Region<'r, 'c, 't> for DetachedRegion<'c, 't> {
    unsafe fn to_c_api(&self) -> MlirRegion {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

impl<'r, 'c: 'r, 't: 'c, R: Region<'r, 'c, 't>> PartialEq<R> for DetachedRegion<'c, 't> {
    fn eq(&self, other: &R) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        // Note that this function only checks for whether the two region handles point to the same underlying region.
        // It does not perform a deep comparison of the contents of these regions.
        unsafe { mlirRegionEqual(self.handle, other.to_c_api()) }
    }
}

impl<'c, 't> Eq for DetachedRegion<'c, 't> {}

impl Drop for DetachedRegion<'_, '_> {
    fn drop(&mut self) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            mlirRegionDestroy(self.handle);
        }
    }
}

impl<'c, 't> From<DetachedBlock<'c, 't>> for DetachedRegion<'c, 't> {
    fn from(value: DetachedBlock<'c, 't>) -> Self {
        let mut region = value.context().region();
        region.append_block(value);
        region
    }
}

impl<'c, 't> FromWithContext<'c, 't, Vec<DetachedBlock<'c, 't>>> for DetachedRegion<'c, 't> {
    fn from_with_context(value: Vec<DetachedBlock<'c, 't>>, context: &'c Context<'t>) -> Self {
        let mut region = context.region();
        for block in value {
            region.append_block(block);
        }
        region
    }
}

impl<'t> Context<'t> {
    /// Creates a new (and empty) [`DetachedRegion`] associated with this [`Context`].
    pub fn region<'c>(&'c self) -> DetachedRegion<'c, 't> {
        DetachedRegion { handle: unsafe { mlirRegionCreate() }, context: self }
    }
}

/// Reference to an MLIR [`Region`] that is owned by an [`Operation`](crate::Operation).
///
/// Note that there are multiple separate lifetime parameters: one for the lifetime of the owner of this [`Region`]
/// reference, `'o`, one for the [`Context`] which is associated with it, `'c`, and one for the lifetime of the thread
/// pool used by that [`Context`], `'t`.
#[derive(Copy, Clone, Debug)]
pub struct RegionRef<'o, 'c: 'o, 't: 'c> {
    /// Handle that represents this [`Region`] reference in the MLIR C API.
    handle: MlirRegion,

    /// [`Context`] associated with this [`Region`] reference.
    context: &'c Context<'t>,

    /// [`PhantomData`] used to track the lifetime of the [`Operation`](crate::Operation)
    /// that owns the underlying [`Region`].
    owner: PhantomData<&'o ()>,
}

impl<'o, 'c, 't> RegionRef<'o, 'c, 't> {
    /// Constructs a new [`RegionRef`] from the provided [`MlirRegion`] handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirRegion, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context, owner: PhantomData }) }
    }
}

impl<'o, 'r: 'o, 'c: 'r, 't: 'c> Region<'r, 'c, 't> for RegionRef<'o, 'c, 't> {
    unsafe fn to_c_api(&self) -> MlirRegion {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

impl<'o, 'r: 'o, 'c: 'r, 't: 'c, R: Region<'r, 'c, 't>> PartialEq<R> for RegionRef<'o, 'c, 't> {
    fn eq(&self, other: &R) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        // Note that this function only checks for whether the two region handles point to the same underlying region.
        // It does not perform a deep comparison of the contents of these regions.
        unsafe { mlirRegionEqual(self.handle, other.to_c_api()) }
    }
}

impl<'o, 'c, 't> Eq for RegionRef<'o, 'c, 't> {}

impl<'r, 'c, 't> From<&'r DetachedRegion<'c, 't>> for RegionRef<'r, 'c, 't> {
    fn from(value: &'r DetachedRegion<'c, 't>) -> Self {
        value.as_ref()
    }
}

/// [`Iterator`] over references of the [`Block`]s contained in a [`Region`].
///
/// Note that this iterator does not hold a borrowed reference to the underlying [`Context`] because that would make it
/// impossible to perform mutating operations on that context (e.g., from within [`Pass`](crate::Pass)es) while
/// iterating over the contents of this iterator.
pub struct RegionBlockRefIterator<'r, 'c: 'r, 't: 'c> {
    /// Current [`BlockRef`] in this iterator (i.e., the [`BlockRef`] that will be returned in the next call to
    /// [`RegionBlockRefIterator::next`]). [`BlockRef`]s are stored in such a way in MLIR that we can always obtain
    /// the next [`Block`] in a [`Region`] given a [`BlockRef`] in that same [`Region`]s.
    current_block: Option<BlockRef<'r, 'c, 't>>,
}

impl<'r, 'c, 't> Iterator for RegionBlockRefIterator<'r, 'c, 't> {
    type Item = BlockRef<'r, 'c, 't>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.current_block.take();
        self.current_block = item.as_ref().and_then(|block| unsafe {
            BlockRef::from_c_api(mlirBlockGetNextInRegion(block.to_c_api()), block.context())
        });
        item
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region() {
        let context = Context::new();
        let mut region_0 = context.region();
        assert!(region_0.is_empty());
        assert_eq!(region_0.context(), &context);
        assert!(region_0.blocks().next().is_none());
        let block_0 = region_0.append_block(context.block_with_no_arguments());
        let block_1 = region_0.append_block(context.block_with_no_arguments());
        let block_2 = region_0.append_block(context.block_with_no_arguments());
        let block_3 = region_0.insert_block(context.block_with_no_arguments(), 1);
        let block_4 = region_0.insert_block(context.block_with_no_arguments(), 0);
        let block_5 = region_0.insert_block_after(context.block_with_no_arguments(), Some(&block_1));
        let block_6 = region_0.insert_block_after(context.block_with_no_arguments(), None);
        let block_7 = region_0.insert_block_before(context.block_with_no_arguments(), Some(&block_4));
        let block_8 = region_0.insert_block_before(context.block_with_no_arguments(), None);
        assert!(!region_0.is_empty());
        assert_eq!(
            region_0.blocks().collect::<Vec<_>>(),
            vec![block_6, block_7, block_4, block_0, block_3, block_1, block_5, block_2, block_8],
        );

        // Test null pointer edge case.
        let bad_handle = MlirRegion { ptr: std::ptr::null_mut() };
        let region = unsafe { RegionRef::from_c_api(bad_handle, &context) };
        assert!(region.is_none());
    }

    #[test]
    fn test_region_take_body() {
        let context = Context::new();
        let mut region_0 = context.region();
        let mut region_1 = context.region();
        region_0.append_block(context.block_with_no_arguments());
        region_0.append_block(context.block_with_no_arguments());
        region_1.append_block(context.block_with_no_arguments());
        assert_eq!(region_0.blocks().count(), 2);
        assert_eq!(region_1.blocks().count(), 1);
        region_1.take_body(region_0.as_ref());
        assert_eq!(region_0.blocks().count(), 0);
        assert_eq!(region_1.blocks().count(), 2);
    }

    #[test]
    fn test_region_equality() {
        let context = Context::new();
        let region_0 = context.region();
        let region_1 = context.region();
        assert_eq!(region_0, region_0);
        assert_eq!(region_0, region_0.as_ref());
        assert_eq!(region_0.as_ref().clone(), region_0.as_ref());
        assert_ne!(region_0, region_1);
        assert_ne!(region_1.as_ref(), region_0);
        assert_eq!(RegionRef::from(&region_1), region_1);
    }

    #[test]
    fn test_region_debug() {
        let context = Context::new();
        let region = context.region();
        assert!(format!("{:?}", region).contains("DetachedRegion"));
        assert!(format!("{:?}", region.as_ref()).contains("RegionRef"));
    }

    #[test]
    fn test_region_from_block() {
        let context = Context::new();
        let block = context.block_with_no_arguments();
        let region = DetachedRegion::from(block);
        assert_eq!(region.blocks().count(), 1);
    }

    #[test]
    fn test_region_from_blocks() {
        let context = Context::new();
        let blocks = vec![];
        let region = DetachedRegion::from_with_context(blocks, &context);
        assert_eq!(region.blocks().count(), 0);
        let blocks = vec![
            context.block_with_no_arguments(),
            context.block_with_no_arguments(),
            context.block_with_no_arguments(),
        ];
        let region = DetachedRegion::from_with_context(blocks, &context);
        assert_eq!(region.blocks().count(), 3);
    }
}
