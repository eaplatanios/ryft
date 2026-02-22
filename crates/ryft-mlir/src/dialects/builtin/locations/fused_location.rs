use ryft_xla_sys::bindings::{
    MlirLocation, mlirLocationFusedGet, mlirLocationFusedGetLocations, mlirLocationFusedGetMetadata,
    mlirLocationFusedGetNumLocations, mlirLocationFusedGetTypeID,
};

use crate::{Attribute, AttributeRef, Context, Location, LocationRef, TypeId, mlir_subtype_trait_impls};

/// [`FusedLocationRef`] are a special kind of [`Location`] that represents multiple underlying locations merged together.
/// When MLIR transformations happen (e.g., inlining, fusion, canonicalization, lowering, etc.), a single IR operation
/// might correspond to multiple different source locations. For example:
/// 
///  - Inlining a function call: the call site has a location, and the callee body has its own locations.
///    The inlined operation corresponds to both.
///  - Loop unrolling: one operation in the original IR produces multiple cloned operations, each with both the
///    original operation's location and the iteration context.
/// 
/// Rather than losing information, MLIR records both by "fusing" them into a [`FusedLocationRef`]. Refer to the
/// [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#fusedloc) for more information.
#[derive(Copy, Clone)]
pub struct FusedLocationRef<'c, 't> {
    /// Handle that represents this [`Location`] in the MLIR C API.
    handle: MlirLocation,

    /// [`Context`] that owns this [`Location`].
    context: &'c Context<'t>,
}

impl<'c, 't> FusedLocationRef<'c, 't> {
    /// Returns the [`TypeId`] that corresponds to [`FusedLocationRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirLocationFusedGetTypeID()).unwrap() }
    }

    /// Returns the number of fused [`Location`]s in this [`FusedLocationRef`].
    pub fn fused_location_count(&self) -> usize {
        unsafe { mlirLocationFusedGetNumLocations(self.handle) as usize }
    }

    /// Returns the fused [`Location`]s in this [`FusedLocationRef`].
    pub fn fused_locations(&self) -> impl Iterator<Item = LocationRef<'c, 't>> {
        unsafe {
            let count = self.fused_location_count();
            let mut buffer: Vec<MlirLocation> = Vec::with_capacity(count);
            mlirLocationFusedGetLocations(self.handle, buffer.as_mut_ptr());
            buffer.set_len(count);
            buffer.into_iter().map(|location| LocationRef::from_c_api(location, self.context).unwrap())
        }
    }

    /// Returns the (optional) metadata of this [`FusedLocationRef`].
    pub fn fused_metadata(&self) -> Option<AttributeRef<'c, 't>> {
        unsafe { AttributeRef::from_c_api(mlirLocationFusedGetMetadata(self.handle), self.context) }
    }
}

mlir_subtype_trait_impls!(FusedLocationRef<'c, 't> as Location, mlir_type = Location, mlir_subtype = Fused);

impl<'t> Context<'t> {
    /// Creates a new [`FusedLocationRef`] owned by this [`Context`].
    pub fn fused_location<'c, L: Location<'c, 't>, A: Attribute<'c, 't>>(
        &'c self,
        locations: &[L],
        attribute: A,
    ) -> FusedLocationRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            let locations = locations.iter().map(|location| location.to_c_api()).collect::<Vec<_>>();
            FusedLocationRef::from_c_api(
                mlirLocationFusedGet(
                    *self.handle.borrow(),
                    locations.len().cast_signed(),
                    locations.as_ptr() as *const _,
                    attribute.to_c_api(),
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

    use crate::locations::tests::{test_location_casting, test_location_display_and_debug};

    use super::*;

    #[test]
    fn test_fused_location_type_id() {
        let fused_location_type_id = FusedLocationRef::type_id();
        assert_eq!(FusedLocationRef::type_id(), FusedLocationRef::type_id());
        assert_eq!(fused_location_type_id, FusedLocationRef::type_id());
    }

    #[test]
    fn test_fused_location() {
        let context = Context::new();
        let location_1 = context.file_location("file1.rs", 10, 5);
        let location_2 = context.file_location("file2.rs", 20, 10);
        let location_3 = context.file_location("file3.rs", 30, 15);
        let metadata = context.unit_attribute();
        let location = context.fused_location(&[location_1, location_2, location_3], metadata);
        assert_eq!(&context, location.context());
        assert_eq!(location.fused_location_count(), 3);
        assert_eq!(location.fused_locations().collect::<Vec<_>>(), vec![location_1, location_2, location_3]);
        assert!(location.fused_metadata().is_some());
    }

    #[test]
    fn test_fused_location_equality() {
        let context = Context::new();
        let file_location_1 = context.file_location("file1.rs", 10, 5);
        let file_location_2 = context.file_location("file2.rs", 20, 10);
        let file_location_3 = context.file_location("file3.rs", 30, 15);
        let metadata = context.unit_attribute();

        // Same locations from the same context must be equal because they are "uniqued".
        let location_1 = context.fused_location(&[file_location_1, file_location_2], metadata);
        let location_2 = context.fused_location(&[file_location_1, file_location_2], metadata);
        assert_eq!(location_1, location_2);

        // Different locations from the same context must not be equal.
        let location_2 = context.fused_location(&[file_location_1, file_location_3], metadata);
        assert_ne!(location_1, location_2);

        // Same locations from different contexts must not be equal.
        let context = Context::new();
        let file_location_1 = context.file_location("file1.rs", 10, 5);
        let file_location_2 = context.file_location("file2.rs", 20, 10);
        let other_metadata = context.unit_attribute();
        let location_2 = context.fused_location(&[file_location_1, file_location_2], other_metadata);
        assert_ne!(location_1, location_2);
    }

    #[test]
    fn test_fused_location_display_and_debug() {
        let context = Context::new();
        let location_1 = context.file_location("test1.rs", 10, 5);
        let location_2 = context.file_location("test2.rs", 20, 10);
        let metadata = context.unit_attribute();
        let location = context.fused_location(&[location_1, location_2], metadata.as_ref());
        test_location_display_and_debug(location, "loc(fused<unit>[\"test1.rs\":10:5, \"test2.rs\":20:10])");
    }

    #[test]
    fn test_fused_location_casting() {
        let context = Context::new();
        let location_1 = context.file_location("file1.rs", 10, 5);
        let location_2 = context.file_location("file2.rs", 20, 10);
        let metadata = context.unit_attribute();
        let location = context.fused_location(&[location_1, location_2], metadata.as_ref());
        test_location_casting(location);
    }
}
