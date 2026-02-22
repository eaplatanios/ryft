use ryft_xla_sys::bindings::{
    MlirAttribute, sdyAttributeIsADimMappingAttr, sdyAttributeIsADimensionShardingAttr, sdyAttributeIsAManualAxesAttr,
    sdyAttributeIsAMeshAttr, sdyAttributeIsAMeshAxisAttr, sdyAttributeIsAOpShardingRuleAttr,
    sdyAttributeIsASubAxisInfoAttr, sdyAttributeIsATensorMappingAttr, sdyAttributeIsATensorShardingAttr,
    sdyAttributeIsATensorShardingPerValueAttr, sdyAttributeIsAnAxisRefAttr, sdyAxisRefAttrGet, sdyAxisRefAttrGetName,
    sdyAxisRefAttrGetSubAxisInfo, sdyDimMappingAttrGet, sdyDimMappingAttrGetFactorIndicesElem,
    sdyDimMappingAttrGetFactorIndicesSize, sdyDimensionShardingAttrGet, sdyDimensionShardingAttrGetAxesElem,
    sdyDimensionShardingAttrGetAxesSize, sdyDimensionShardingAttrGetIsClosed, sdyDimensionShardingAttrGetPriority,
    sdyManualAxesAttrGet, sdyManualAxesAttrGetAxesElem, sdyManualAxesAttrGetAxesSize, sdyMeshAttrGet,
    sdyMeshAttrGetAxesElem, sdyMeshAttrGetAxesSize, sdyMeshAttrGetDeviceIdsElem, sdyMeshAttrGetDeviceIdsSize,
    sdyMeshAxisAttrGet, sdyMeshAxisAttrGetName, sdyMeshAxisAttrGetSize, sdyOpShardingRuleAttrGet,
    sdyOpShardingRuleAttrGetBlockedPropagationFactorsElem, sdyOpShardingRuleAttrGetBlockedPropagationFactorsSize,
    sdyOpShardingRuleAttrGetFactorSizesElem, sdyOpShardingRuleAttrGetFactorSizesSize, sdyOpShardingRuleAttrGetIsCustom,
    sdyOpShardingRuleAttrGetNeedReplicationFactorsElem, sdyOpShardingRuleAttrGetNeedReplicationFactorsSize,
    sdyOpShardingRuleAttrGetOperandMappingsElem, sdyOpShardingRuleAttrGetOperandMappingsSize,
    sdyOpShardingRuleAttrGetPermutationFactorsElem, sdyOpShardingRuleAttrGetPermutationFactorsSize,
    sdyOpShardingRuleAttrGetReductionFactorsElem, sdyOpShardingRuleAttrGetReductionFactorsSize,
    sdyOpShardingRuleAttrGetResultMappingsElem, sdyOpShardingRuleAttrGetResultMappingsSize, sdySubAxisInfoAttrGet,
    sdySubAxisInfoAttrGetPreSize, sdySubAxisInfoAttrGetSize, sdyTensorMappingAttrGet,
    sdyTensorMappingAttrGetDimMappingsElem, sdyTensorMappingAttrGetDimMappingsSize, sdyTensorMappingAttrGetRank,
    sdyTensorShardingAttrGet, sdyTensorShardingAttrGetDimShardingsElem, sdyTensorShardingAttrGetDimShardingsSize,
    sdyTensorShardingAttrGetMeshOrRef, sdyTensorShardingAttrGetReplicatedAxesElem,
    sdyTensorShardingAttrGetReplicatedAxesSize, sdyTensorShardingAttrGetUnreducedAxesElem,
    sdyTensorShardingAttrGetUnreducedAxesSize, sdyTensorShardingPerValueAttrGet,
    sdyTensorShardingPerValueAttrGetShardingsElem, sdyTensorShardingPerValueAttrGetShardingsSize,
};

use crate::{Attribute, AttributeRef, Context, DialectHandle, StringRef, mlir_subtype_trait_impls};

/// Shardy [`Attribute`] for `all_to_all` parameter entries.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#alltoallparamattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct AllToAllParamAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for AllToAllParamAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        if attribute_has_shardy_mnemonic(attribute, "all_to_all_param") { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(AllToAllParamAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Shardy [`Attribute`] for lists of `all_to_all` parameter entries.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#alltoallparamlistattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct AllToAllParamListAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for AllToAllParamListAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        if attribute_has_shardy_mnemonic(attribute, "all_to_all_param_list") {
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

mlir_subtype_trait_impls!(AllToAllParamListAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Shardy [`Attribute`] referencing a full axis or a split sub-axis.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#axisrefattr)
/// for more information.
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
        unsafe { StringRef::from_c_api(sdyAxisRefAttrGetName(self.handle)) }
    }

    /// Returns split metadata when this references a sub-axis.
    pub fn sub_axis_info(&self) -> Option<SubAxisInfoAttributeRef<'c, 't>> {
        unsafe { SubAxisInfoAttributeRef::from_c_api(sdyAxisRefAttrGetSubAxisInfo(self.handle), self.context) }
    }
}

impl<'c, 't> Attribute<'c, 't> for AxisRefAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { sdyAttributeIsAnAxisRefAttr(handle) } {
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

impl<'t> Context<'t> {
    /// Creates a Shardy [`AxisRefAttributeRef`].
    ///
    /// # Parameters
    ///
    ///   - `name`: Axis name to reference.
    ///   - `sub_axis_info`: Optional split metadata for sub-axis references.
    pub fn shardy_axis_ref<'c, N: AsRef<str>>(
        &'c self,
        name: N,
        sub_axis_info: Option<SubAxisInfoAttributeRef<'c, 't>>,
    ) -> AxisRefAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::shardy());
        unsafe {
            AxisRefAttributeRef::from_c_api(
                sdyAxisRefAttrGet(
                    *self.handle.borrow(),
                    StringRef::from(name.as_ref()).to_c_api(),
                    sub_axis_info.map(|value| value.to_c_api()).unwrap_or(self.null_attribute().to_c_api()),
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Shardy [`Attribute`] for lists of [`AxisRefAttributeRef`] values.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#axisreflistattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct AxisRefListAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for AxisRefListAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        if attribute_has_shardy_mnemonic(attribute, "axis_ref_list") { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(AxisRefListAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Shardy [`Attribute`] mapping one tensor dimension to sharding-rule factors.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#dimmappingattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct DimMappingAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> DimMappingAttributeRef<'c, 't> {
    /// Returns factor indices this dimension maps to.
    pub fn factor_indices(&self) -> Vec<usize> {
        unsafe {
            let count = sdyDimMappingAttrGetFactorIndicesSize(self.handle).cast_unsigned();
            let mut indices = Vec::with_capacity(count);
            for index in 0..count {
                indices.push(sdyDimMappingAttrGetFactorIndicesElem(self.handle, index.cast_signed()) as usize);
            }
            indices
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for DimMappingAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { sdyAttributeIsADimMappingAttr(handle) } {
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

mlir_subtype_trait_impls!(DimMappingAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a Shardy [`DimMappingAttributeRef`].
    ///
    /// # Parameters
    ///
    ///   - `factor_indices`: Rule-factor indices mapped to a tensor dimension.
    pub fn shardy_dim_mapping<'c>(&'c self, factor_indices: &[usize]) -> DimMappingAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::shardy());
        unsafe {
            let factor_indices = factor_indices.iter().map(|value| *value as i64).collect::<Vec<_>>();
            DimMappingAttributeRef::from_c_api(
                sdyDimMappingAttrGet(
                    *self.handle.borrow(),
                    factor_indices.len().cast_signed(),
                    factor_indices.as_ptr(),
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Shardy [`Attribute`] describing sharding for one tensor dimension.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#dimensionshardingattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct DimensionShardingAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> DimensionShardingAttributeRef<'c, 't> {
    /// Returns axis refs used to shard this dimension.
    pub fn axes(&self) -> Vec<AxisRefAttributeRef<'c, 't>> {
        unsafe {
            let count = sdyDimensionShardingAttrGetAxesSize(self.handle).cast_unsigned();
            let mut axes = Vec::with_capacity(count);
            for index in 0..count {
                axes.push(
                    AxisRefAttributeRef::from_c_api(
                        sdyDimensionShardingAttrGetAxesElem(self.handle, index.cast_signed()),
                        self.context,
                    )
                    .unwrap(),
                );
            }
            axes
        }
    }

    /// Returns whether this dimension is closed to additional sharding.
    pub fn is_closed(&self) -> bool {
        unsafe { sdyDimensionShardingAttrGetIsClosed(self.handle) }
    }

    /// Returns optional user-provided sharding priority.
    pub fn priority(&self) -> Option<usize> {
        let priority = unsafe { sdyDimensionShardingAttrGetPriority(self.handle) };
        if priority < 0 { None } else { Some(priority as usize) }
    }
}

impl<'c, 't> Attribute<'c, 't> for DimensionShardingAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { sdyAttributeIsADimensionShardingAttr(handle) } {
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

mlir_subtype_trait_impls!(DimensionShardingAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a Shardy [`DimensionShardingAttributeRef`].
    ///
    /// # Parameters
    ///
    ///   - `axes`: Axis refs used to shard one tensor dimension.
    ///   - `is_closed`: Whether the dimension is closed to additional sharding.
    ///   - `priority`: Optional user-defined priority.
    pub fn shardy_dimension_sharding<'c>(
        &'c self,
        axes: &[AxisRefAttributeRef<'c, 't>],
        is_closed: bool,
        priority: Option<usize>,
    ) -> DimensionShardingAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::shardy());
        unsafe {
            let axes = axes.iter().map(|axis| axis.to_c_api()).collect::<Vec<_>>();
            DimensionShardingAttributeRef::from_c_api(
                sdyDimensionShardingAttrGet(
                    *self.handle.borrow(),
                    axes.len().cast_signed(),
                    axes.as_ptr(),
                    is_closed,
                    priority.map(|value| value as i64).unwrap_or(-1),
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Shardy [`Attribute`] for lists of axis-ref lists.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#listofaxisreflistsattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct ListOfAxisRefListsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for ListOfAxisRefListsAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        if attribute_has_shardy_mnemonic(attribute, "list_of_axis_ref_lists") {
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

mlir_subtype_trait_impls!(ListOfAxisRefListsAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Shardy [`Attribute`] listing manual axes for `sdy.manual_computation`.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#manualaxesattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct ManualAxesAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> ManualAxesAttributeRef<'c, 't> {
    /// Returns manual axis names.
    pub fn axes(&self) -> Vec<StringRef<'c>> {
        unsafe {
            let count = sdyManualAxesAttrGetAxesSize(self.handle).cast_unsigned();
            let mut axes = Vec::with_capacity(count);
            for index in 0..count {
                axes.push(StringRef::from_c_api(sdyManualAxesAttrGetAxesElem(self.handle, index.cast_signed())));
            }
            axes
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for ManualAxesAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { sdyAttributeIsAManualAxesAttr(handle) } {
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

mlir_subtype_trait_impls!(ManualAxesAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a Shardy [`ManualAxesAttributeRef`].
    ///
    /// # Parameters
    ///
    ///   - `axes`: Axes that are handled manually inside `sdy.manual_computation`.
    pub fn shardy_manual_axes<'c, A: AsRef<str>>(&'c self, axes: &[A]) -> ManualAxesAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::shardy());
        unsafe {
            let axes = axes.iter().map(|axis| self.string_attribute(axis.as_ref()).to_c_api()).collect::<Vec<_>>();
            ManualAxesAttributeRef::from_c_api(
                sdyManualAxesAttrGet(*self.handle.borrow(), axes.len().cast_signed(), axes.as_ptr()),
                self,
            )
            .unwrap()
        }
    }
}

/// Shardy [`Attribute`] representing a logical device mesh.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#meshattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct MeshAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> MeshAttributeRef<'c, 't> {
    /// Returns mesh axes in declaration order.
    pub fn axes(&self) -> Vec<MeshAxisAttributeRef<'c, 't>> {
        unsafe {
            let count = sdyMeshAttrGetAxesSize(self.handle).cast_unsigned();
            let mut axes = Vec::with_capacity(count);
            for index in 0..count {
                axes.push(
                    MeshAxisAttributeRef::from_c_api(
                        sdyMeshAttrGetAxesElem(self.handle, index.cast_signed()),
                        self.context,
                    )
                    .unwrap(),
                );
            }
            axes
        }
    }

    /// Returns device IDs associated with this mesh.
    pub fn device_ids(&self) -> Vec<usize> {
        unsafe {
            let count = sdyMeshAttrGetDeviceIdsSize(self.handle).cast_unsigned() as usize;
            let mut device_ids = Vec::with_capacity(count);
            for index in 0..count {
                device_ids.push(sdyMeshAttrGetDeviceIdsElem(self.handle, index as i64) as usize);
            }
            device_ids
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for MeshAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { sdyAttributeIsAMeshAttr(handle) } {
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
    /// Creates a Shardy [`MeshAttributeRef`].
    ///
    /// # Parameters
    ///
    ///   - `axes`: Mesh axes in declaration order.
    ///   - `device_ids`: Optional explicit device ordering for the mesh.
    pub fn shardy_mesh<'c>(
        &'c self,
        axes: &[MeshAxisAttributeRef<'c, 't>],
        device_ids: &[usize],
    ) -> MeshAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::shardy());
        unsafe {
            let axes = axes.iter().map(|axis| axis.to_c_api()).collect::<Vec<_>>();
            let device_ids = device_ids.iter().map(|value| *value as i64).collect::<Vec<_>>();
            MeshAttributeRef::from_c_api(
                sdyMeshAttrGet(
                    *self.handle.borrow(),
                    axes.len().cast_signed(),
                    axes.as_ptr(),
                    device_ids.len().cast_signed(),
                    device_ids.as_ptr(),
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Shardy [`Attribute`] representing a single mesh axis.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#meshaxisattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct MeshAxisAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> MeshAxisAttributeRef<'c, 't> {
    /// Returns the mesh-axis name.
    pub fn name(&self) -> StringRef<'c> {
        unsafe { StringRef::from_c_api(sdyMeshAxisAttrGetName(self.handle)) }
    }

    /// Returns the mesh-axis size.
    pub fn size(&self) -> usize {
        unsafe { sdyMeshAxisAttrGetSize(self.handle) as usize }
    }
}

impl<'c, 't> Attribute<'c, 't> for MeshAxisAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { sdyAttributeIsAMeshAxisAttr(handle) } {
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

impl<'t> Context<'t> {
    /// Creates a Shardy [`MeshAxisAttributeRef`].
    ///
    /// # Parameters
    ///
    ///   - `name`: Logical axis name.
    ///   - `size`: Positive axis size.
    pub fn shardy_mesh_axis<'c, N: AsRef<str>>(&'c self, name: N, size: usize) -> MeshAxisAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::shardy());
        unsafe {
            MeshAxisAttributeRef::from_c_api(
                sdyMeshAxisAttrGet(*self.handle.borrow(), StringRef::from(name.as_ref()).to_c_api(), size as i64),
                self,
            )
            .unwrap()
        }
    }
}

/// Shardy [`Attribute`] describing sharding propagation behavior for an operation.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#opshardingruleattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct OpShardingRuleAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> OpShardingRuleAttributeRef<'c, 't> {
    /// Returns whether this rule is marked custom by the producer.
    pub fn is_custom_rule(&self) -> bool {
        unsafe { sdyOpShardingRuleAttrGetIsCustom(self.handle) }
    }

    /// Returns factor sizes in rule-factor order.
    pub fn factor_sizes(&self) -> Vec<usize> {
        unsafe {
            let count = sdyOpShardingRuleAttrGetFactorSizesSize(self.handle).cast_unsigned();
            let mut values = Vec::with_capacity(count);
            for index in 0..count {
                values.push(sdyOpShardingRuleAttrGetFactorSizesElem(self.handle, index.cast_signed()) as usize);
            }
            values
        }
    }

    /// Returns tensor mappings for operation operands.
    pub fn operand_mappings(&self) -> Vec<TensorMappingAttributeRef<'c, 't>> {
        unsafe {
            let count = sdyOpShardingRuleAttrGetOperandMappingsSize(self.handle).cast_unsigned();
            let mut values = Vec::with_capacity(count);
            for index in 0..count {
                values.push(
                    TensorMappingAttributeRef::from_c_api(
                        sdyOpShardingRuleAttrGetOperandMappingsElem(self.handle, index.cast_signed()),
                        self.context,
                    )
                    .unwrap(),
                );
            }
            values
        }
    }

    /// Returns tensor mappings for operation results.
    pub fn result_mappings(&self) -> Vec<TensorMappingAttributeRef<'c, 't>> {
        unsafe {
            let count = sdyOpShardingRuleAttrGetResultMappingsSize(self.handle).cast_unsigned();
            let mut values = Vec::with_capacity(count);
            for index in 0..count {
                values.push(
                    TensorMappingAttributeRef::from_c_api(
                        sdyOpShardingRuleAttrGetResultMappingsElem(self.handle, index.cast_signed()),
                        self.context,
                    )
                    .unwrap(),
                );
            }
            values
        }
    }

    /// Returns factor indices used as reduction factors.
    pub fn reduction_factors(&self) -> Vec<usize> {
        unsafe {
            let count = sdyOpShardingRuleAttrGetReductionFactorsSize(self.handle).cast_unsigned();
            let mut values = Vec::with_capacity(count);
            for index in 0..count {
                values.push(sdyOpShardingRuleAttrGetReductionFactorsElem(self.handle, index.cast_signed()) as usize);
            }
            values
        }
    }

    /// Returns factor indices that require full replication.
    pub fn need_replication_factors(&self) -> Vec<usize> {
        unsafe {
            let count = sdyOpShardingRuleAttrGetNeedReplicationFactorsSize(self.handle).cast_unsigned();
            let mut values = Vec::with_capacity(count);
            for index in 0..count {
                values.push(sdyOpShardingRuleAttrGetNeedReplicationFactorsElem(self.handle, index.cast_signed()) as usize);
            }
            values
        }
    }

    /// Returns factor indices used for permutation behavior.
    pub fn permutation_factors(&self) -> Vec<usize> {
        unsafe {
            let count = sdyOpShardingRuleAttrGetPermutationFactorsSize(self.handle).cast_unsigned();
            let mut values = Vec::with_capacity(count);
            for index in 0..count {
                values.push(sdyOpShardingRuleAttrGetPermutationFactorsElem(self.handle, index.cast_signed()) as usize);
            }
            values
        }
    }

    /// Returns factor indices blocked for propagation.
    pub fn blocked_propagation_factors(&self) -> Vec<usize> {
        unsafe {
            let count = sdyOpShardingRuleAttrGetBlockedPropagationFactorsSize(self.handle).cast_unsigned();
            let mut values = Vec::with_capacity(count);
            for index in 0..count {
                values
                    .push(sdyOpShardingRuleAttrGetBlockedPropagationFactorsElem(self.handle, index.cast_signed())
                        as usize);
            }
            values
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for OpShardingRuleAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { sdyAttributeIsAOpShardingRuleAttr(handle) } {
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

mlir_subtype_trait_impls!(OpShardingRuleAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a Shardy [`OpShardingRuleAttributeRef`].
    ///
    /// # Parameters
    ///
    ///   - `factor_sizes`: Rule-factor sizes.
    ///   - `operand_mappings`: Per-operand tensor mappings.
    ///   - `result_mappings`: Per-result tensor mappings.
    ///   - `reduction_factors`: Reduction factor indices.
    ///   - `need_replication_factors`: Replication-required factor indices.
    ///   - `permutation_factors`: Permutation factor indices.
    ///   - `blocked_propagation_factors`: Propagation-blocked factor indices.
    ///   - `is_custom_rule`: Whether the rule is marked custom.
    pub fn shardy_op_sharding_rule<'c>(
        &'c self,
        factor_sizes: &[usize],
        operand_mappings: &[TensorMappingAttributeRef<'c, 't>],
        result_mappings: &[TensorMappingAttributeRef<'c, 't>],
        reduction_factors: &[usize],
        need_replication_factors: &[usize],
        permutation_factors: &[usize],
        blocked_propagation_factors: &[usize],
        is_custom_rule: bool,
    ) -> OpShardingRuleAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::shardy());
        unsafe {
            let factor_sizes = factor_sizes.iter().map(|value| *value as i64).collect::<Vec<_>>();
            let operand_mappings = operand_mappings.iter().map(|value| value.to_c_api()).collect::<Vec<_>>();
            let result_mappings = result_mappings.iter().map(|value| value.to_c_api()).collect::<Vec<_>>();
            let reduction_factors = reduction_factors.iter().map(|value| *value as i64).collect::<Vec<_>>();
            let need_replication_factors =
                need_replication_factors.iter().map(|value| *value as i64).collect::<Vec<_>>();
            let permutation_factors = permutation_factors.iter().map(|value| *value as i64).collect::<Vec<_>>();
            let blocked_propagation_factors =
                blocked_propagation_factors.iter().map(|value| *value as i64).collect::<Vec<_>>();
            OpShardingRuleAttributeRef::from_c_api(
                sdyOpShardingRuleAttrGet(
                    *self.handle.borrow(),
                    factor_sizes.len().cast_signed(),
                    factor_sizes.as_ptr(),
                    operand_mappings.len().cast_signed(),
                    operand_mappings.as_ptr(),
                    result_mappings.len().cast_signed(),
                    result_mappings.as_ptr(),
                    reduction_factors.len().cast_signed(),
                    reduction_factors.as_ptr(),
                    need_replication_factors.len().cast_signed(),
                    need_replication_factors.as_ptr(),
                    permutation_factors.len().cast_signed(),
                    permutation_factors.as_ptr(),
                    blocked_propagation_factors.len().cast_signed(),
                    blocked_propagation_factors.as_ptr(),
                    is_custom_rule,
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Shardy [`Attribute`] describing a split sub-axis.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#subaxisinfoattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct SubAxisInfoAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> SubAxisInfoAttributeRef<'c, 't> {
    /// Returns the `pre_size` term.
    pub fn pre_size(&self) -> usize {
        unsafe { sdySubAxisInfoAttrGetPreSize(self.handle) as usize }
    }

    /// Returns the split-factor `size` term.
    pub fn size(&self) -> usize {
        unsafe { sdySubAxisInfoAttrGetSize(self.handle) as usize }
    }
}

impl<'c, 't> Attribute<'c, 't> for SubAxisInfoAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { sdyAttributeIsASubAxisInfoAttr(handle) } {
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

impl<'t> Context<'t> {
    /// Creates a Shardy [`SubAxisInfoAttributeRef`].
    ///
    /// # Parameters
    ///
    ///   - `pre_size`: Product of split factors to the left.
    ///   - `size`: Split-factor size for this sub-axis.
    pub fn shardy_sub_axis_info<'c>(&'c self, pre_size: usize, size: usize) -> SubAxisInfoAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::shardy());
        unsafe {
            SubAxisInfoAttributeRef::from_c_api(
                sdySubAxisInfoAttrGet(*self.handle.borrow(), pre_size as i64, size as i64),
                self,
            )
            .unwrap()
        }
    }
}

/// Shardy [`Attribute`] carrying all dimension mappings for one tensor.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#tensormappingattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct TensorMappingAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> TensorMappingAttributeRef<'c, 't> {
    /// Returns the tensor rank represented by this mapping.
    pub fn rank(&self) -> usize {
        unsafe { sdyTensorMappingAttrGetRank(self.handle).cast_unsigned() }
    }

    /// Returns dimension mappings in major-to-minor dimension order.
    pub fn dim_mappings(&self) -> Vec<DimMappingAttributeRef<'c, 't>> {
        unsafe {
            let count = sdyTensorMappingAttrGetDimMappingsSize(self.handle).cast_unsigned();
            let mut dim_mappings = Vec::with_capacity(count);
            for index in 0..count {
                dim_mappings.push(
                    DimMappingAttributeRef::from_c_api(
                        sdyTensorMappingAttrGetDimMappingsElem(self.handle, index.cast_signed()),
                        self.context,
                    )
                    .unwrap(),
                );
            }
            dim_mappings
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for TensorMappingAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { sdyAttributeIsATensorMappingAttr(handle) } {
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

mlir_subtype_trait_impls!(TensorMappingAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a Shardy [`TensorMappingAttributeRef`].
    ///
    /// # Parameters
    ///
    ///   - `dim_mappings`: One dimension mapping entry per tensor dimension.
    pub fn shardy_tensor_mapping<'c>(
        &'c self,
        dim_mappings: &[DimMappingAttributeRef<'c, 't>],
    ) -> TensorMappingAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::shardy());
        unsafe {
            let dim_mappings = dim_mappings.iter().map(|value| value.to_c_api()).collect::<Vec<_>>();
            TensorMappingAttributeRef::from_c_api(
                sdyTensorMappingAttrGet(*self.handle.borrow(), dim_mappings.len().cast_signed(), dim_mappings.as_ptr()),
                self,
            )
            .unwrap()
        }
    }
}

/// Shardy [`Attribute`] encoding complete tensor sharding metadata.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#tensorshardingattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct TensorShardingAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> TensorShardingAttributeRef<'c, 't> {
    /// Returns the mesh payload or mesh symbol reference.
    pub fn mesh_or_ref(&self) -> AttributeRef<'c, 't> {
        unsafe { AttributeRef::from_c_api(sdyTensorShardingAttrGetMeshOrRef(self.handle), self.context).unwrap() }
    }

    /// Returns per-dimension sharding descriptors.
    pub fn dim_shardings(&self) -> Vec<DimensionShardingAttributeRef<'c, 't>> {
        unsafe {
            let count = sdyTensorShardingAttrGetDimShardingsSize(self.handle).cast_unsigned();
            let mut dim_shardings = Vec::with_capacity(count);
            for index in 0..count {
                dim_shardings.push(
                    DimensionShardingAttributeRef::from_c_api(
                        sdyTensorShardingAttrGetDimShardingsElem(self.handle, index.cast_signed()),
                        self.context,
                    )
                    .unwrap(),
                );
            }
            dim_shardings
        }
    }

    /// Returns axes explicitly marked as replicated.
    pub fn replicated_axes(&self) -> Vec<AxisRefAttributeRef<'c, 't>> {
        unsafe {
            let count = sdyTensorShardingAttrGetReplicatedAxesSize(self.handle).cast_unsigned();
            let mut axes = Vec::with_capacity(count);
            for index in 0..count {
                axes.push(
                    AxisRefAttributeRef::from_c_api(
                        sdyTensorShardingAttrGetReplicatedAxesElem(self.handle, index.cast_signed()),
                        self.context,
                    )
                    .unwrap(),
                );
            }
            axes
        }
    }

    /// Returns axes explicitly marked as unreduced.
    pub fn unreduced_axes(&self) -> Vec<AxisRefAttributeRef<'c, 't>> {
        unsafe {
            let count = sdyTensorShardingAttrGetUnreducedAxesSize(self.handle).cast_unsigned();
            let mut axes = Vec::with_capacity(count);
            for index in 0..count {
                axes.push(
                    AxisRefAttributeRef::from_c_api(
                        sdyTensorShardingAttrGetUnreducedAxesElem(self.handle, index.cast_signed()),
                        self.context,
                    )
                    .unwrap(),
                );
            }
            axes
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for TensorShardingAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { sdyAttributeIsATensorShardingAttr(handle) } {
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

mlir_subtype_trait_impls!(TensorShardingAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a Shardy [`TensorShardingAttributeRef`].
    ///
    /// # Parameters
    ///
    ///   - `mesh_or_ref`: Mesh payload or mesh symbol reference.
    ///   - `dim_shardings`: Per-dimension sharding descriptors.
    ///   - `replicated_axes`: Axes explicitly marked replicated.
    ///   - `unreduced_axes`: Axes explicitly marked unreduced.
    pub fn shardy_tensor_sharding<'c, M: Attribute<'c, 't>>(
        &'c self,
        mesh_or_ref: M,
        dim_shardings: &[DimensionShardingAttributeRef<'c, 't>],
        replicated_axes: &[AxisRefAttributeRef<'c, 't>],
        unreduced_axes: &[AxisRefAttributeRef<'c, 't>],
    ) -> TensorShardingAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::shardy());
        unsafe {
            let dim_shardings = dim_shardings.iter().map(|value| value.to_c_api()).collect::<Vec<_>>();
            let replicated_axes = replicated_axes.iter().map(|value| value.to_c_api()).collect::<Vec<_>>();
            let unreduced_axes = unreduced_axes.iter().map(|value| value.to_c_api()).collect::<Vec<_>>();
            TensorShardingAttributeRef::from_c_api(
                sdyTensorShardingAttrGet(
                    *self.handle.borrow(),
                    mesh_or_ref.to_c_api(),
                    dim_shardings.len().cast_signed(),
                    dim_shardings.as_ptr(),
                    replicated_axes.len().cast_signed(),
                    replicated_axes.as_ptr(),
                    unreduced_axes.len().cast_signed(),
                    unreduced_axes.as_ptr(),
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Shardy [`Attribute`] carrying a [`TensorShardingAttributeRef`] per value slot.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#tensorshardingpervalueattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct TensorShardingPerValueAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> TensorShardingPerValueAttributeRef<'c, 't> {
    /// Returns one tensor sharding entry per value slot.
    pub fn shardings(&self) -> Vec<TensorShardingAttributeRef<'c, 't>> {
        unsafe {
            let count = sdyTensorShardingPerValueAttrGetShardingsSize(self.handle).cast_unsigned();
            let mut shardings = Vec::with_capacity(count);
            for index in 0..count {
                shardings.push(
                    TensorShardingAttributeRef::from_c_api(
                        sdyTensorShardingPerValueAttrGetShardingsElem(self.handle, index.cast_signed()),
                        self.context,
                    )
                    .unwrap(),
                );
            }
            shardings
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for TensorShardingPerValueAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { sdyAttributeIsATensorShardingPerValueAttr(handle) } {
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

mlir_subtype_trait_impls!(TensorShardingPerValueAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a Shardy [`TensorShardingPerValueAttributeRef`].
    ///
    /// # Parameters
    ///
    ///   - `shardings`: One sharding entry per value slot.
    pub fn shardy_tensor_sharding_per_value<'c>(
        &'c self,
        shardings: &[TensorShardingAttributeRef<'c, 't>],
    ) -> TensorShardingPerValueAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::shardy());
        unsafe {
            let shardings = shardings.iter().map(|value| value.to_c_api()).collect::<Vec<_>>();
            TensorShardingPerValueAttributeRef::from_c_api(
                sdyTensorShardingPerValueAttrGet(
                    *self.handle.borrow(),
                    shardings.len().cast_signed(),
                    shardings.as_ptr(),
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Internal helper that checks whether the textual form of `attribute` has the `#sdy.<mnemonic>` prefix
/// with a mnemonic token exactly equal to `mnemonic`. This is needed because the StableHLO C API does not
/// expose type checking functions for some of the Shardy attributes that we need to support.
fn attribute_has_shardy_mnemonic<'c, 't: 'c, A: Attribute<'c, 't>, M: AsRef<str>>(attribute: A, mnemonic: M) -> bool {
    let rendered = attribute.to_string();
    let Some(suffix) = rendered.strip_prefix("#sdy.") else {
        return false;
    };
    let end = suffix
        .char_indices()
        .find_map(|(index, character)| {
            if character == ':'
                || character == '>'
                || character == '<'
                || character == ' '
                || character == '\t'
                || character == '\n'
            {
                Some(index)
            } else {
                None
            }
        })
        .unwrap_or(suffix.len());
    &suffix[..end] == mnemonic.as_ref()
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_all_to_all_param_attribute() {
        let context = Context::new();
        let attribute = context
            .opaque_attribute("sdy", "all_to_all_param", context.none_type())
            .as_ref()
            .cast::<AllToAllParamAttributeRef>()
            .unwrap();
        assert_eq!(&context, attribute.context());
        test_attribute_display_and_debug(attribute, "#sdy.all_to_all_param");
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_all_to_all_param_list_attribute() {
        let context = Context::new();
        let attribute = context
            .opaque_attribute("sdy", "all_to_all_param_list", context.none_type())
            .as_ref()
            .cast::<AllToAllParamListAttributeRef>()
            .unwrap();
        assert_eq!(&context, attribute.context());
        test_attribute_display_and_debug(attribute, "#sdy.all_to_all_param_list");
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_axis_ref_attribute() {
        let context = Context::new();
        let sub_axis_info = context.shardy_sub_axis_info(2, 4);
        let attribute = context.shardy_axis_ref("data", Some(sub_axis_info));
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.name().as_str().unwrap(), "data");
        assert_eq!(attribute.sub_axis_info(), Some(sub_axis_info));
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_axis_ref_list_attribute() {
        let context = Context::new();
        let attribute = context
            .opaque_attribute("sdy", "axis_ref_list", context.none_type())
            .as_ref()
            .cast::<AxisRefListAttributeRef>()
            .unwrap();
        assert_eq!(&context, attribute.context());
        test_attribute_display_and_debug(attribute, "#sdy.axis_ref_list");
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_dim_mapping_attribute() {
        let context = Context::new();
        let attribute = context.shardy_dim_mapping(&[0, 1, 3]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.factor_indices(), vec![0, 1, 3]);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_dimension_sharding_attribute() {
        let context = Context::new();
        let axis_ref = context.shardy_axis_ref("data", None);
        let attribute = context.shardy_dimension_sharding(&[axis_ref], true, Some(2));
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.axes(), vec![axis_ref]);
        assert!(attribute.is_closed());
        assert_eq!(attribute.priority(), Some(2));
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_list_of_axis_ref_lists_attribute() {
        let context = Context::new();
        let attribute = context
            .opaque_attribute("sdy", "list_of_axis_ref_lists", context.none_type())
            .as_ref()
            .cast::<ListOfAxisRefListsAttributeRef>()
            .unwrap();
        assert_eq!(&context, attribute.context());
        test_attribute_display_and_debug(attribute, "#sdy.list_of_axis_ref_lists");
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_manual_axes_attribute() {
        let context = Context::new();
        let attribute = context.shardy_manual_axes(&["data", "model"]);

        assert_eq!(&context, attribute.context());
        assert_eq!(
            attribute.axes().iter().map(|axis| axis.as_str().unwrap()).collect::<Vec<_>>(),
            vec!["data", "model"],
        );
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_mesh_attribute() {
        let context = Context::new();
        let axis_a = context.shardy_mesh_axis("a", 2);
        let axis_b = context.shardy_mesh_axis("b", 4);
        let attribute = context.shardy_mesh(&[axis_a, axis_b], &[0, 2, 4, 6, 1, 3, 5, 7]);

        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.axes(), vec![axis_a, axis_b]);
        assert_eq!(attribute.device_ids(), vec![0, 2, 4, 6, 1, 3, 5, 7]);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_mesh_axis_attribute() {
        let context = Context::new();
        let attribute = context.shardy_mesh_axis("data", 8);

        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.name().as_str().unwrap(), "data");
        assert_eq!(attribute.size(), 8);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_op_sharding_rule_attribute() {
        let context = Context::new();
        let dim_mapping = context.shardy_dim_mapping(&[0, 1]);
        let tensor_mapping = context.shardy_tensor_mapping(&[dim_mapping]);
        let attribute = context.shardy_op_sharding_rule(
            &[8, 16],
            &[tensor_mapping],
            &[tensor_mapping],
            &[0],
            &[1],
            &[0],
            &[],
            true,
        );
        assert_eq!(&context, attribute.context());
        assert!(attribute.is_custom_rule());
        assert_eq!(attribute.factor_sizes(), vec![8, 16]);
        assert_eq!(attribute.operand_mappings(), vec![tensor_mapping]);
        assert_eq!(attribute.result_mappings(), vec![tensor_mapping]);
        assert_eq!(attribute.reduction_factors(), vec![0]);
        assert_eq!(attribute.need_replication_factors(), vec![1]);
        assert_eq!(attribute.permutation_factors(), vec![0]);
        assert!(attribute.blocked_propagation_factors().is_empty());
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_sub_axis_info_attribute() {
        let context = Context::new();
        let attribute = context.shardy_sub_axis_info(3, 5);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.pre_size(), 3);
        assert_eq!(attribute.size(), 5);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_tensor_mapping_attribute() {
        let context = Context::new();
        let dim_mapping_0 = context.shardy_dim_mapping(&[0]);
        let dim_mapping_1 = context.shardy_dim_mapping(&[1]);
        let attribute = context.shardy_tensor_mapping(&[dim_mapping_0, dim_mapping_1]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.rank(), 2);
        assert_eq!(attribute.dim_mappings(), vec![dim_mapping_0, dim_mapping_1]);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_tensor_sharding_attribute() {
        let context = Context::new();
        let mesh_axis = context.shardy_mesh_axis("x", 2);
        let mesh = context.shardy_mesh(&[mesh_axis], &[]);
        let axis_ref = context.shardy_axis_ref("x", None);
        let dim_sharding = context.shardy_dimension_sharding(&[axis_ref], true, None);
        let attribute = context.shardy_tensor_sharding(mesh, &[dim_sharding], &[axis_ref], &[]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.mesh_or_ref(), mesh.as_ref());
        assert_eq!(attribute.dim_shardings(), vec![dim_sharding]);
        assert_eq!(attribute.replicated_axes(), vec![axis_ref]);
        assert!(attribute.unreduced_axes().is_empty());
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_tensor_sharding_per_value_attribute() {
        let context = Context::new();
        let mesh_axis = context.shardy_mesh_axis("x", 2);
        let mesh = context.shardy_mesh(&[mesh_axis], &[]);
        let axis_ref = context.shardy_axis_ref("x", None);
        let dim_sharding = context.shardy_dimension_sharding(&[axis_ref], true, None);
        let tensor_sharding = context.shardy_tensor_sharding(mesh, &[dim_sharding], &[], &[]);
        let attribute = context.shardy_tensor_sharding_per_value(&[tensor_sharding]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.shardings(), vec![tensor_sharding]);
        test_attribute_casting(attribute);
    }
}
