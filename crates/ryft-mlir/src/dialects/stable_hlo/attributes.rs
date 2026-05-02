use ryft_xla_sys::bindings::{
    MlirAttribute, mlirShapedTypeGetDynamicSize, stablehloAttributeIsAMeshAttr, stablehloAttributeIsAMeshAxisAttr,
    stablehloAttributeIsAReplicaGroupMeshAxesAttr, stablehloAttributeIsASubAxisInfoAttr,
    stablehloAttributeIsAnAxisRefAttr, stablehloAttributeIsTypeExtensions, stablehloAxisRefAttrGet,
    stablehloAxisRefAttrGetName, stablehloAxisRefAttrGetSubAxisInfo, stablehloMeshAttrGet, stablehloMeshAttrGetAxes,
    stablehloMeshAttrGetDeviceIds, stablehloMeshAxisAttrGet, stablehloMeshAxisAttrGetName,
    stablehloMeshAxisAttrGetSize, stablehloReplicaGroupMeshAxesAttrGet, stablehloReplicaGroupMeshAxesAttrGetAxes,
    stablehloReplicaGroupMeshAxesAttrGetMesh, stablehloSubAxisInfoAttrGet, stablehloSubAxisInfoAttrGetPreSize,
    stablehloSubAxisInfoAttrGetSize, stablehloTypeExtensionsGet, stablehloTypeExtensionsGetBoundsElem,
    stablehloTypeExtensionsGetBoundsSize,
};

use crate::{
    ArrayAttributeRef, Attribute, AttributeRef, Context, DenseIntegerElementsAttributeRef, DialectHandle, StringRef,
    mlir_subtype_trait_impls,
};

/// StableHLO [`Attribute`] that is used to extend the built-in MLIR [`TensorTypeRef`](crate::TensorTypeRef) with
/// StableHLO tensor-specific properties. These properties are not modeled in the built-in MLIR type. This is included
/// in [`TensorTypeRef`](crate::TensorTypeRef) for StableHLO types via its
/// [`TensorTypeRef::encoding`](crate::TensorTypeRef::encoding) attribute.
#[derive(Copy, Clone)]
pub struct TensorTypeExtensionsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> TensorTypeExtensionsAttributeRef<'c, 't> {
    /// Returns the bounds for the dimensions of the associated [`TensorTypeRef`](crate::TensorTypeRef). The returned
    /// vector a length equal to the number of dimensions of the associated [`TensorTypeRef`](crate::TensorTypeRef)
    /// (i.e., equal to its _rank_). For each dimension, it contains either a bound on its size if it is a dimension
    /// with a [`Size::Dynamic`](crate::Size::Dynamic) size, or [`None`] if it has either a
    /// [`Size::Static`](crate::Size::Static) size, or a [`Size::Dynamic`](crate::Size::Dynamic)
    /// size and no bound specified for it.
    pub fn bounds(&self) -> Vec<Option<usize>> {
        unsafe {
            let count = stablehloTypeExtensionsGetBoundsSize(self.handle).cast_unsigned();
            let mut bounds = Vec::with_capacity(count);
            for i in 0..count {
                let bound = stablehloTypeExtensionsGetBoundsElem(self.handle, i.cast_signed());
                bounds.push(if bound == mlirShapedTypeGetDynamicSize() { None } else { Some(bound as usize) });
            }
            bounds
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for TensorTypeExtensionsAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { stablehloAttributeIsTypeExtensions(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(TensorTypeExtensionsAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a new StableHLO [`TensorTypeExtensionsAttributeRef`] owned by this [`Context`].
    ///
    /// Refer to the documentation of [`TensorTypeExtensionsAttributeRef::bounds`] for information on the `bounds`
    /// argument of this function.
    pub fn stable_hlo_tensor_type_extensions<'c>(
        &'c self,
        bounds: &[Option<usize>],
    ) -> TensorTypeExtensionsAttributeRef<'c, 't> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo());
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            let bounds = bounds
                .iter()
                .map(|bound| match bound {
                    None => mlirShapedTypeGetDynamicSize(),
                    Some(bound) => *bound as i64,
                })
                .collect::<Vec<_>>();
            TensorTypeExtensionsAttributeRef::from_c_api(
                stablehloTypeExtensionsGet(*self.handle.borrow(), bounds.len().cast_signed(), bounds.as_ptr()),
                self,
            )
            .unwrap()
        }
    }
}

/// StableHLO [`Attribute`] that identifies a contiguous sub-axis derived from a full mesh axis.
#[derive(Copy, Clone)]
pub struct SubAxisInfoAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> SubAxisInfoAttributeRef<'c, 't> {
    /// Returns the product of the sizes of the sub-axes that appear before this sub-axis.
    pub fn pre_size(&self) -> i64 {
        unsafe { stablehloSubAxisInfoAttrGetPreSize(self.handle) }
    }

    /// Returns the size of this sub-axis.
    pub fn size(&self) -> i64 {
        unsafe { stablehloSubAxisInfoAttrGetSize(self.handle) }
    }
}

impl<'c, 't> Attribute<'c, 't> for SubAxisInfoAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { stablehloAttributeIsASubAxisInfoAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(SubAxisInfoAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// StableHLO [`Attribute`] that references either a full mesh axis or a split sub-axis.
#[derive(Copy, Clone)]
pub struct AxisRefAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> AxisRefAttributeRef<'c, 't> {
    /// Returns the referenced axis name.
    pub fn name(&self) -> StringRef<'c> {
        unsafe { StringRef::from_c_api(stablehloAxisRefAttrGetName(self.handle)) }
    }

    /// Returns split metadata when this references a sub-axis.
    pub fn sub_axis_info(&self) -> Option<SubAxisInfoAttributeRef<'c, 't>> {
        unsafe { SubAxisInfoAttributeRef::from_c_api(stablehloAxisRefAttrGetSubAxisInfo(self.handle), self.context) }
    }
}

impl<'c, 't> Attribute<'c, 't> for AxisRefAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { stablehloAttributeIsAnAxisRefAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(AxisRefAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// StableHLO [`Attribute`] that represents replica groups using a mesh and referenced mesh axes.
#[derive(Copy, Clone)]
pub struct ReplicaGroupMeshAxesAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> ReplicaGroupMeshAxesAttributeRef<'c, 't> {
    /// Returns the mesh attribute, which may be a symbol reference or an inline mesh.
    pub fn mesh(&self) -> AttributeRef<'c, 't> {
        unsafe {
            AttributeRef::from_c_api(stablehloReplicaGroupMeshAxesAttrGetMesh(self.handle), self.context)
                .expect("invalid StableHLO replica-group mesh")
        }
    }

    /// Returns the array of axes used to form replica groups.
    pub fn axes(&self) -> ArrayAttributeRef<'c, 't> {
        unsafe {
            ArrayAttributeRef::from_c_api(stablehloReplicaGroupMeshAxesAttrGetAxes(self.handle), self.context)
                .expect("invalid StableHLO replica-group axes")
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for ReplicaGroupMeshAxesAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { stablehloAttributeIsAReplicaGroupMeshAxesAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(ReplicaGroupMeshAxesAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// StableHLO [`Attribute`] that defines a single named mesh axis and its size.
#[derive(Copy, Clone)]
pub struct MeshAxisAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> MeshAxisAttributeRef<'c, 't> {
    /// Returns the mesh axis name.
    pub fn name(&self) -> StringRef<'c> {
        unsafe { StringRef::from_c_api(stablehloMeshAxisAttrGetName(self.handle)) }
    }

    /// Returns the mesh axis size.
    pub fn size(&self) -> i64 {
        unsafe { stablehloMeshAxisAttrGetSize(self.handle) }
    }
}

impl<'c, 't> Attribute<'c, 't> for MeshAxisAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { stablehloAttributeIsAMeshAxisAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(MeshAxisAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// StableHLO [`Attribute`] that defines an inline device mesh.
#[derive(Copy, Clone)]
pub struct MeshAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> MeshAttributeRef<'c, 't> {
    /// Returns the array of mesh axis attributes.
    pub fn axes(&self) -> ArrayAttributeRef<'c, 't> {
        unsafe {
            ArrayAttributeRef::from_c_api(stablehloMeshAttrGetAxes(self.handle), self.context)
                .expect("invalid StableHLO mesh axes")
        }
    }

    /// Returns the optional dense device-id tensor for this mesh.
    pub fn device_ids(&self) -> Option<DenseIntegerElementsAttributeRef<'c, 't>> {
        unsafe {
            DenseIntegerElementsAttributeRef::from_c_api(stablehloMeshAttrGetDeviceIds(self.handle), self.context)
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for MeshAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { stablehloAttributeIsAMeshAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(MeshAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a new StableHLO [`SubAxisInfoAttributeRef`] owned by this [`Context`].
    pub fn stable_hlo_sub_axis_info<'c>(&'c self, pre_size: i64, size: i64) -> SubAxisInfoAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::stable_hlo());
        unsafe {
            SubAxisInfoAttributeRef::from_c_api(
                stablehloSubAxisInfoAttrGet(*self.handle.borrow(), pre_size, size),
                self,
            )
            .unwrap()
        }
    }

    /// Creates a new StableHLO [`AxisRefAttributeRef`] owned by this [`Context`].
    pub fn stable_hlo_axis_ref<'c, N: AsRef<str>>(
        &'c self,
        name: N,
        sub_axis_info: Option<SubAxisInfoAttributeRef<'c, 't>>,
    ) -> AxisRefAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::stable_hlo());
        unsafe {
            AxisRefAttributeRef::from_c_api(
                stablehloAxisRefAttrGet(
                    *self.handle.borrow(),
                    StringRef::from(name.as_ref()).to_c_api(),
                    sub_axis_info.map(|value| value.to_c_api()).unwrap_or(self.null_attribute().to_c_api()),
                ),
                self,
            )
            .unwrap()
        }
    }

    /// Creates a new StableHLO [`ReplicaGroupMeshAxesAttributeRef`] owned by this [`Context`].
    pub fn stable_hlo_replica_group_mesh_axes<'c, M: Attribute<'c, 't>>(
        &'c self,
        mesh: M,
        axes: ArrayAttributeRef<'c, 't>,
    ) -> ReplicaGroupMeshAxesAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::stable_hlo());
        unsafe {
            ReplicaGroupMeshAxesAttributeRef::from_c_api(
                stablehloReplicaGroupMeshAxesAttrGet(*self.handle.borrow(), mesh.to_c_api(), axes.to_c_api()),
                self,
            )
            .unwrap()
        }
    }

    /// Creates a new StableHLO [`MeshAxisAttributeRef`] owned by this [`Context`].
    pub fn stable_hlo_mesh_axis<'c, N: AsRef<str>>(&'c self, name: N, size: i64) -> MeshAxisAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::stable_hlo());
        unsafe {
            MeshAxisAttributeRef::from_c_api(
                stablehloMeshAxisAttrGet(*self.handle.borrow(), StringRef::from(name.as_ref()).to_c_api(), size),
                self,
            )
            .unwrap()
        }
    }

    /// Creates a new StableHLO [`MeshAttributeRef`] owned by this [`Context`].
    pub fn stable_hlo_mesh<'c>(
        &'c self,
        axes: ArrayAttributeRef<'c, 't>,
        device_ids: Option<DenseIntegerElementsAttributeRef<'c, 't>>,
    ) -> MeshAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::stable_hlo());
        unsafe {
            MeshAttributeRef::from_c_api(
                stablehloMeshAttrGet(
                    *self.handle.borrow(),
                    axes.to_c_api(),
                    device_ids.map(|value| value.to_c_api()).unwrap_or(self.null_attribute().to_c_api()),
                ),
                self,
            )
            .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_tensor_type_extensions_attribute() {
        let context = Context::new();
        let bounds = vec![Some(10), None, Some(20), None];
        let attribute = context.stable_hlo_tensor_type_extensions(&bounds);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.bounds(), bounds);
    }

    #[test]
    fn test_tensor_type_extensions_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_tensor_type_extensions(&[Some(10), None, Some(20), None]);
        let attribute_2 = context.stable_hlo_tensor_type_extensions(&[Some(10), None, Some(20), None]);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_tensor_type_extensions(&[None, None, Some(20)]);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_tensor_type_extensions(&[Some(10), None, Some(20), None]);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_tensor_type_extensions_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_tensor_type_extensions(&[Some(10), None, Some(20), None]);
        test_attribute_display_and_debug(attribute, "#stablehlo.bounds<10, ?, 20, ?>");
    }

    #[test]
    fn test_tensor_type_extensions_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_tensor_type_extensions(&[Some(10), None, Some(20), None]);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_sub_axis_info_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_sub_axis_info(2, 4);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.pre_size(), 2);
        assert_eq!(attribute.size(), 4);
    }

    #[test]
    fn test_sub_axis_info_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_sub_axis_info(2, 4);
        let attribute_2 = context.stable_hlo_sub_axis_info(2, 4);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_sub_axis_info(1, 4);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_sub_axis_info(2, 4);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_sub_axis_info_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_sub_axis_info(2, 4);
        test_attribute_display_and_debug(attribute, "#stablehlo<sub_axis_info(2)4>");
    }

    #[test]
    fn test_sub_axis_info_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_sub_axis_info(2, 4);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_axis_ref_attribute() {
        let context = Context::new();
        let sub_axis_info = context.stable_hlo_sub_axis_info(2, 4);
        let attribute = context.stable_hlo_axis_ref("x", Some(sub_axis_info));
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.name().as_str().unwrap(), "x");
        assert_eq!(attribute.sub_axis_info(), Some(sub_axis_info));

        let attribute = context.stable_hlo_axis_ref("y", None);
        assert_eq!(attribute.name().as_str().unwrap(), "y");
        assert_eq!(attribute.sub_axis_info(), None);
    }

    #[test]
    fn test_axis_ref_attribute_equality() {
        let context = Context::new();
        let sub_axis_info = context.stable_hlo_sub_axis_info(2, 4);

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_axis_ref("x", Some(sub_axis_info));
        let attribute_2 = context.stable_hlo_axis_ref("x", Some(sub_axis_info));
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_axis_ref("y", Some(sub_axis_info));
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let sub_axis_info = context.stable_hlo_sub_axis_info(2, 4);
        let attribute_2 = context.stable_hlo_axis_ref("x", Some(sub_axis_info));
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_axis_ref_attribute_display_and_debug() {
        let context = Context::new();
        let sub_axis_info = context.stable_hlo_sub_axis_info(2, 4);
        let attribute = context.stable_hlo_axis_ref("x", Some(sub_axis_info));
        test_attribute_display_and_debug(attribute, "#stablehlo.axis_ref<name = \"x\", sub_axis_info = (2)4>");
    }

    #[test]
    fn test_axis_ref_attribute_casting() {
        let context = Context::new();
        let sub_axis_info = context.stable_hlo_sub_axis_info(2, 4);
        let attribute = context.stable_hlo_axis_ref("x", Some(sub_axis_info));
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_replica_group_mesh_axes_attribute() {
        let context = Context::new();
        let mesh = context.flat_symbol_ref_attribute("mesh");
        let axis_x = context.stable_hlo_axis_ref("x", Some(context.stable_hlo_sub_axis_info(2, 4)));
        let axis_y = context.stable_hlo_axis_ref("y", None);
        let axes = context.array_attribute(&[axis_x, axis_y]);
        let attribute = context.stable_hlo_replica_group_mesh_axes(mesh, axes);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.mesh(), mesh.as_ref());
        assert_eq!(attribute.axes(), axes);
    }

    #[test]
    fn test_replica_group_mesh_axes_attribute_equality() {
        let context = Context::new();
        let mesh = context.flat_symbol_ref_attribute("mesh");
        let axis_x = context.stable_hlo_axis_ref("x", None);
        let axis_y = context.stable_hlo_axis_ref("y", None);
        let axes = context.array_attribute(&[axis_x, axis_y]);

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_replica_group_mesh_axes(mesh, axes);
        let attribute_2 = context.stable_hlo_replica_group_mesh_axes(mesh, axes);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let axes = context.array_attribute(&[axis_x]);
        let attribute_2 = context.stable_hlo_replica_group_mesh_axes(mesh, axes);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let mesh = context.flat_symbol_ref_attribute("mesh");
        let axis_x = context.stable_hlo_axis_ref("x", None);
        let axis_y = context.stable_hlo_axis_ref("y", None);
        let axes = context.array_attribute(&[axis_x, axis_y]);
        let attribute_2 = context.stable_hlo_replica_group_mesh_axes(mesh, axes);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_replica_group_mesh_axes_attribute_display_and_debug() {
        let context = Context::new();
        let mesh = context.flat_symbol_ref_attribute("mesh");
        let axis_x = context.stable_hlo_axis_ref("x", Some(context.stable_hlo_sub_axis_info(2, 4)));
        let axis_y = context.stable_hlo_axis_ref("y", None);
        let axes = context.array_attribute(&[axis_x, axis_y]);
        let attribute = context.stable_hlo_replica_group_mesh_axes(mesh, axes);
        test_attribute_display_and_debug(
            attribute,
            "#stablehlo.replica_group_mesh_axes<mesh = @mesh, axes = \
             [#stablehlo.axis_ref<name = \"x\", sub_axis_info = (2)4>, #stablehlo.axis_ref<name = \"y\">]>",
        );
    }

    #[test]
    fn test_replica_group_mesh_axes_attribute_casting() {
        let context = Context::new();
        let mesh = context.flat_symbol_ref_attribute("mesh");
        let axis = context.stable_hlo_axis_ref("x", None);
        let axes = context.array_attribute(&[axis]);
        let attribute = context.stable_hlo_replica_group_mesh_axes(mesh, axes);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_mesh_axis_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_mesh_axis("x", 2);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.name().as_str().unwrap(), "x");
        assert_eq!(attribute.size(), 2);
    }

    #[test]
    fn test_mesh_axis_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_mesh_axis("x", 2);
        let attribute_2 = context.stable_hlo_mesh_axis("x", 2);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_mesh_axis("y", 2);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_mesh_axis("x", 2);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_mesh_axis_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_mesh_axis("x", 2);
        test_attribute_display_and_debug(attribute, "#stablehlo.mesh_axis<name = \"x\", size = 2>");
    }

    #[test]
    fn test_mesh_axis_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_mesh_axis("x", 2);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_mesh_attribute() {
        let context = Context::new();
        let axis_x = context.stable_hlo_mesh_axis("x", 2);
        let axis_y = context.stable_hlo_mesh_axis("y", 4);
        let axes = context.array_attribute(&[axis_x, axis_y]);
        let attribute = context.stable_hlo_mesh(axes, None);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.axes(), axes);
        assert_eq!(attribute.device_ids(), None);
    }

    #[test]
    fn test_mesh_attribute_equality() {
        let context = Context::new();
        let axis_x = context.stable_hlo_mesh_axis("x", 2);
        let axis_y = context.stable_hlo_mesh_axis("y", 4);
        let axes = context.array_attribute(&[axis_x, axis_y]);

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_mesh(axes, None);
        let attribute_2 = context.stable_hlo_mesh(axes, None);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let axes = context.array_attribute(&[axis_x]);
        let attribute_2 = context.stable_hlo_mesh(axes, None);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let axis_x = context.stable_hlo_mesh_axis("x", 2);
        let axis_y = context.stable_hlo_mesh_axis("y", 4);
        let axes = context.array_attribute(&[axis_x, axis_y]);
        let attribute_2 = context.stable_hlo_mesh(axes, None);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_mesh_attribute_display_and_debug() {
        let context = Context::new();
        let axis_x = context.stable_hlo_mesh_axis("x", 2);
        let axis_y = context.stable_hlo_mesh_axis("y", 4);
        let axes = context.array_attribute(&[axis_x, axis_y]);
        let attribute = context.stable_hlo_mesh(axes, None);
        test_attribute_display_and_debug(
            attribute,
            "#stablehlo.mesh<axes=[<name = \"x\", size = 2>, <name = \"y\", size = 4>]>",
        );
    }

    #[test]
    fn test_mesh_attribute_casting() {
        let context = Context::new();
        let axis = context.stable_hlo_mesh_axis("x", 2);
        let axes = context.array_attribute(&[axis]);
        let attribute = context.stable_hlo_mesh(axes, None);
        test_attribute_casting(attribute);
    }
}
