// TODO(eaplatanios): Clean this up and make sure it is correct.

use ryft_xla_sys::bindings::{
    MlirAttribute, mlirGPUObjectAttrGet, mlirGPUObjectAttrGetFormat, mlirGPUObjectAttrGetKernels,
    mlirGPUObjectAttrGetObject, mlirGPUObjectAttrGetProperties, mlirGPUObjectAttrGetTarget,
    mlirGPUObjectAttrGetWithKernels, mlirGPUObjectAttrHasKernels, mlirGPUObjectAttrHasProperties,
};

use crate::{Attribute, AttributeRef, Context, DialectHandle, OpaqueAttributeRef, StringRef, mlir_subtype_trait_impls};

/// MLIR [`Attribute`] wrapper for attributes that belong to the `gpu` dialect namespace.
#[derive(Copy, Clone)]
pub struct GpuAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> GpuAttributeRef<'c, 't> {
    /// Returns the attribute mnemonic if it can be derived from its textual form.
    pub fn mnemonic(&self) -> Option<String> {
        attribute_mnemonic(*self)
    }
}

impl<'c, 't> Attribute<'c, 't> for GpuAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        if attribute_is_gpu(attribute) { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(GpuAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Enumerates GPU address spaces used by `gpu.address_space` attributes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AddressSpace {
    /// Global device memory address space.
    Global,
    /// Workgroup/shared address space.
    Workgroup,
    /// Private address space.
    Private,
}

impl AddressSpace {
    /// Returns the MLIR string representation of this [`AddressSpace`].
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Workgroup => "workgroup",
            Self::Private => "private",
        }
    }
}

impl<'s> TryFrom<&'s str> for AddressSpace {
    type Error = String;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        match value {
            "global" => Ok(Self::Global),
            "workgroup" => Ok(Self::Workgroup),
            "private" => Ok(Self::Private),
            _ => Err(format!("'{value}' is not a valid gpu.address_space")),
        }
    }
}

/// MLIR [`Attribute`] that represents a GPU address space.
#[derive(Copy, Clone)]
pub struct AddressSpaceAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> AddressSpaceAttributeRef<'c, 't> {
    /// Returns the [`AddressSpace`] value stored in this attribute.
    pub fn value(&self) -> AddressSpace {
        let rendered = self.to_string();
        let value = rendered
            .strip_prefix("#gpu.address_space<")
            .and_then(|suffix| suffix.strip_suffix('>'))
            .map(ToOwned::to_owned)
            .or_else(|| {
                rendered.strip_prefix("#gpu<").and_then(|suffix| suffix.strip_suffix('>')).and_then(|contents| {
                    let mut tokens = contents.split_whitespace();
                    if tokens.next() == Some("address_space") { tokens.next().map(ToOwned::to_owned) } else { None }
                })
            })
            .ok_or_else(|| format!("invalid gpu.address_space attribute: '{rendered}'"))
            .and_then(|value| AddressSpace::try_from(value.as_str()))
            .unwrap();
        value
    }
}

impl<'c, 't> Attribute<'c, 't> for AddressSpaceAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        let gpu_attribute = attribute.cast::<GpuAttributeRef>()?;
        if gpu_attribute.mnemonic().as_deref() == Some("address_space") { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(AddressSpaceAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Enumerates GPU dimensions (`x`, `y`, `z`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dimension {
    X,
    Y,
    Z,
}

impl Dimension {
    /// Returns the MLIR string representation of this [`Dimension`].
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::X => "x",
            Self::Y => "y",
            Self::Z => "z",
        }
    }
}

impl<'s> TryFrom<&'s str> for Dimension {
    type Error = String;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        match value {
            "x" => Ok(Self::X),
            "y" => Ok(Self::Y),
            "z" => Ok(Self::Z),
            _ => Err(format!("'{value}' is not a valid gpu.dim dimension")),
        }
    }
}

/// MLIR [`Attribute`] that represents a GPU dimension (`#gpu.dim`).
#[derive(Copy, Clone)]
pub struct DimensionAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> DimensionAttributeRef<'c, 't> {
    /// Returns the [`Dimension`] value stored in this attribute.
    pub fn value(&self) -> Dimension {
        let rendered = self.to_string();
        let value = parse_enum_attribute_value(rendered, "dim")
            .and_then(|value| Dimension::try_from(value.as_str()).ok())
            .unwrap();
        value
    }
}

impl<'c, 't> Attribute<'c, 't> for DimensionAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        let gpu_attribute = attribute.cast::<GpuAttributeRef>()?;
        if gpu_attribute.mnemonic().as_deref() == Some("dim") { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(DimensionAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Enumerates reduction operations supported by `gpu.all_reduce` and `gpu.subgroup_reduce`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AllReduceOperation {
    Add,
    Mul,
    MinUi,
    MinSi,
    MinNumF,
    MaxUi,
    MaxSi,
    MaxNumF,
    And,
    Or,
    Xor,
    MinimumF,
    MaximumF,
}

impl AllReduceOperation {
    /// Returns the MLIR string representation of this [`AllReduceOperation`].
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::MinUi => "minui",
            Self::MinSi => "minsi",
            Self::MinNumF => "minnumf",
            Self::MaxUi => "maxui",
            Self::MaxSi => "maxsi",
            Self::MaxNumF => "maxnumf",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::MinimumF => "minimumf",
            Self::MaximumF => "maximumf",
        }
    }
}

impl<'s> TryFrom<&'s str> for AllReduceOperation {
    type Error = String;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        match value {
            "add" => Ok(Self::Add),
            "mul" => Ok(Self::Mul),
            "minui" => Ok(Self::MinUi),
            "minsi" => Ok(Self::MinSi),
            "minnumf" => Ok(Self::MinNumF),
            "maxui" => Ok(Self::MaxUi),
            "maxsi" => Ok(Self::MaxSi),
            "maxnumf" => Ok(Self::MaxNumF),
            "and" => Ok(Self::And),
            "or" => Ok(Self::Or),
            "xor" => Ok(Self::Xor),
            "minimumf" => Ok(Self::MinimumF),
            "maximumf" => Ok(Self::MaximumF),
            _ => Err(format!("'{value}' is not a valid gpu.all_reduce_op")),
        }
    }
}

/// MLIR [`Attribute`] that represents an `gpu.all_reduce_op` enum value.
#[derive(Copy, Clone)]
pub struct AllReduceOperationAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> AllReduceOperationAttributeRef<'c, 't> {
    /// Returns the [`AllReduceOperation`] value stored in this attribute.
    pub fn value(&self) -> AllReduceOperation {
        let rendered = self.to_string();
        let value = parse_enum_attribute_value(rendered, "all_reduce_op")
            .and_then(|value| AllReduceOperation::try_from(value.as_str()).ok())
            .unwrap();
        value
    }
}

impl<'c, 't> Attribute<'c, 't> for AllReduceOperationAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        let gpu_attribute = attribute.cast::<GpuAttributeRef>()?;
        if gpu_attribute.mnemonic().as_deref() == Some("all_reduce_op") { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(AllReduceOperationAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Enumerates shuffle modes supported by `gpu.shuffle`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShuffleMode {
    Xor,
    Down,
    Up,
    Idx,
}

impl ShuffleMode {
    /// Returns the MLIR string representation of this [`ShuffleMode`].
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Xor => "xor",
            Self::Down => "down",
            Self::Up => "up",
            Self::Idx => "idx",
        }
    }
}

impl<'s> TryFrom<&'s str> for ShuffleMode {
    type Error = String;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        match value {
            "xor" => Ok(Self::Xor),
            "down" => Ok(Self::Down),
            "up" => Ok(Self::Up),
            "idx" => Ok(Self::Idx),
            _ => Err(format!("'{value}' is not a valid gpu.shuffle_mode")),
        }
    }
}

/// MLIR [`Attribute`] that represents a `gpu.shuffle_mode` enum value.
#[derive(Copy, Clone)]
pub struct ShuffleModeAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> ShuffleModeAttributeRef<'c, 't> {
    /// Returns the [`ShuffleMode`] value stored in this attribute.
    pub fn value(&self) -> ShuffleMode {
        let rendered = self.to_string();
        let value = parse_enum_attribute_value(rendered, "shuffle_mode")
            .and_then(|value| ShuffleMode::try_from(value.as_str()).ok())
            .unwrap();
        value
    }
}

impl<'c, 't> Attribute<'c, 't> for ShuffleModeAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        let gpu_attribute = attribute.cast::<GpuAttributeRef>()?;
        if gpu_attribute.mnemonic().as_deref() == Some("shuffle_mode") { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(ShuffleModeAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Enumerates MMA elementwise operations supported by `gpu.subgroup_mma_elementwise`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MmaElementwiseOp {
    AddF,
    MulF,
    SubF,
    MaxF,
    MinF,
    DivF,
    AddI,
    MulI,
    SubI,
    DivS,
    DivU,
    NegateF,
    NegateS,
    ExtF,
}

impl MmaElementwiseOp {
    /// Returns the MLIR string representation of this [`MmaElementwiseOp`].
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::AddF => "addf",
            Self::MulF => "mulf",
            Self::SubF => "subf",
            Self::MaxF => "maxf",
            Self::MinF => "minf",
            Self::DivF => "divf",
            Self::AddI => "addi",
            Self::MulI => "muli",
            Self::SubI => "subi",
            Self::DivS => "divs",
            Self::DivU => "divu",
            Self::NegateF => "negatef",
            Self::NegateS => "negates",
            Self::ExtF => "extf",
        }
    }
}

impl<'s> TryFrom<&'s str> for MmaElementwiseOp {
    type Error = String;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        match value {
            "addf" => Ok(Self::AddF),
            "mulf" => Ok(Self::MulF),
            "subf" => Ok(Self::SubF),
            "maxf" => Ok(Self::MaxF),
            "minf" => Ok(Self::MinF),
            "divf" => Ok(Self::DivF),
            "addi" => Ok(Self::AddI),
            "muli" => Ok(Self::MulI),
            "subi" => Ok(Self::SubI),
            "divs" => Ok(Self::DivS),
            "divu" => Ok(Self::DivU),
            "negatef" => Ok(Self::NegateF),
            "negates" => Ok(Self::NegateS),
            "extf" => Ok(Self::ExtF),
            _ => Err(format!("'{value}' is not a valid gpu.mma_element_wise")),
        }
    }
}

/// MLIR [`Attribute`] that represents a `gpu.mma_element_wise` enum value.
#[derive(Copy, Clone)]
pub struct MmaElementwiseOpAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> MmaElementwiseOpAttributeRef<'c, 't> {
    /// Returns the [`MmaElementwiseOp`] value stored in this attribute.
    pub fn value(&self) -> MmaElementwiseOp {
        let rendered = self.to_string();
        let value = parse_enum_attribute_value(rendered, "mma_element_wise")
            .and_then(|value| MmaElementwiseOp::try_from(value.as_str()).ok())
            .unwrap();
        value
    }
}

impl<'c, 't> Attribute<'c, 't> for MmaElementwiseOpAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        let gpu_attribute = attribute.cast::<GpuAttributeRef>()?;
        if gpu_attribute.mnemonic().as_deref() == Some("mma_element_wise") {
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

mlir_subtype_trait_impls!(MmaElementwiseOpAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Enumerates pruning strategies for `gpu.create_2to4_spmat`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Prune2To4SpMatFlag {
    None,
    PruneOnly,
    PruneAndCheck,
}

impl Prune2To4SpMatFlag {
    /// Returns the MLIR string representation of this [`Prune2To4SpMatFlag`].
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::PruneOnly => "PRUNE_ONLY",
            Self::PruneAndCheck => "PRUNE_AND_CHECK",
        }
    }
}

impl<'s> TryFrom<&'s str> for Prune2To4SpMatFlag {
    type Error = String;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        match value {
            "NONE" => Ok(Self::None),
            "PRUNE_ONLY" => Ok(Self::PruneOnly),
            "PRUNE_AND_CHECK" => Ok(Self::PruneAndCheck),
            _ => Err(format!("'{value}' is not a valid gpu.prune_2to4_spmat_flag")),
        }
    }
}

/// MLIR [`Attribute`] that represents a `gpu.prune_2to4_spmat_flag` enum value.
#[derive(Copy, Clone)]
pub struct Prune2To4SpMatFlagAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Prune2To4SpMatFlagAttributeRef<'c, 't> {
    /// Returns the [`Prune2To4SpMatFlag`] value stored in this attribute.
    pub fn value(&self) -> Prune2To4SpMatFlag {
        let rendered = self.to_string();
        let value = parse_enum_attribute_value(rendered, "prune_2to4_spmat_flag")
            .and_then(|value| Prune2To4SpMatFlag::try_from(value.as_str()).ok())
            .unwrap();
        value
    }
}

impl<'c, 't> Attribute<'c, 't> for Prune2To4SpMatFlagAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        let gpu_attribute = attribute.cast::<GpuAttributeRef>()?;
        if gpu_attribute.mnemonic().as_deref() == Some("prune_2to4_spmat_flag") {
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

mlir_subtype_trait_impls!(Prune2To4SpMatFlagAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Enumerates matrix transpose modes used by sparse tensor operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TransposeMode {
    NonTranspose,
    Transpose,
    ConjugateTranspose,
}

impl TransposeMode {
    /// Returns the MLIR string representation of this [`TransposeMode`].
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::NonTranspose => "NON_TRANSPOSE",
            Self::Transpose => "TRANSPOSE",
            Self::ConjugateTranspose => "CONJUGATE_TRANSPOSE",
        }
    }
}

impl<'s> TryFrom<&'s str> for TransposeMode {
    type Error = String;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        match value {
            "NON_TRANSPOSE" => Ok(Self::NonTranspose),
            "TRANSPOSE" => Ok(Self::Transpose),
            "CONJUGATE_TRANSPOSE" => Ok(Self::ConjugateTranspose),
            _ => Err(format!("'{value}' is not a valid gpu.mat_transpose_mode")),
        }
    }
}

/// MLIR [`Attribute`] that represents a `gpu.mat_transpose_mode` enum value.
#[derive(Copy, Clone)]
pub struct TransposeModeAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> TransposeModeAttributeRef<'c, 't> {
    /// Returns the [`TransposeMode`] value stored in this attribute.
    pub fn value(&self) -> TransposeMode {
        let rendered = self.to_string();
        let value = parse_enum_attribute_value(rendered, "mat_transpose_mode")
            .and_then(|value| TransposeMode::try_from(value.as_str()).ok())
            .unwrap();
        value
    }
}

impl<'c, 't> Attribute<'c, 't> for TransposeModeAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        let gpu_attribute = attribute.cast::<GpuAttributeRef>()?;
        if gpu_attribute.mnemonic().as_deref() == Some("mat_transpose_mode") {
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

mlir_subtype_trait_impls!(TransposeModeAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Enumerates whether `gpu.spgemm_work_estimation_or_compute` performs work estimation or compute.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpGemmWorkEstimationOrComputeKind {
    WorkEstimation,
    Compute,
}

impl SpGemmWorkEstimationOrComputeKind {
    /// Returns the MLIR string representation of this [`SpGemmWorkEstimationOrComputeKind`].
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::WorkEstimation => "WORK_ESTIMATION",
            Self::Compute => "COMPUTE",
        }
    }
}

impl<'s> TryFrom<&'s str> for SpGemmWorkEstimationOrComputeKind {
    type Error = String;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        match value {
            "WORK_ESTIMATION" => Ok(Self::WorkEstimation),
            "COMPUTE" => Ok(Self::Compute),
            _ => Err(format!("'{value}' is not a valid gpu.spgemm_work_estimation_or_compute_kind")),
        }
    }
}

/// MLIR [`Attribute`] that represents a `gpu.spgemm_work_estimation_or_compute_kind` enum value.
#[derive(Copy, Clone)]
pub struct SpGemmWorkEstimationOrComputeKindAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> SpGemmWorkEstimationOrComputeKindAttributeRef<'c, 't> {
    /// Returns the [`SpGemmWorkEstimationOrComputeKind`] value stored in this attribute.
    pub fn value(&self) -> SpGemmWorkEstimationOrComputeKind {
        let rendered = self.to_string();
        let value = parse_enum_attribute_value(rendered, "spgemm_work_estimation_or_compute_kind")
            .and_then(|value| SpGemmWorkEstimationOrComputeKind::try_from(value.as_str()).ok())
            .unwrap();
        value
    }
}

impl<'c, 't> Attribute<'c, 't> for SpGemmWorkEstimationOrComputeKindAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        let gpu_attribute = attribute.cast::<GpuAttributeRef>()?;
        if gpu_attribute.mnemonic().as_deref() == Some("spgemm_work_estimation_or_compute_kind") {
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

mlir_subtype_trait_impls!(SpGemmWorkEstimationOrComputeKindAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Enumerates subgroup broadcast modes supported by `gpu.subgroup_broadcast`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BroadcastType {
    FirstActiveLane,
    SpecificLane,
}

impl BroadcastType {
    /// Returns the MLIR string representation of this [`BroadcastType`].
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::FirstActiveLane => "first_active_lane",
            Self::SpecificLane => "specific_lane",
        }
    }
}

impl<'s> TryFrom<&'s str> for BroadcastType {
    type Error = String;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        match value {
            "first_active_lane" => Ok(Self::FirstActiveLane),
            "specific_lane" => Ok(Self::SpecificLane),
            _ => Err(format!("'{value}' is not a valid gpu.broadcast")),
        }
    }
}

/// MLIR [`Attribute`] that represents a `gpu.broadcast` enum value.
#[derive(Copy, Clone)]
pub struct BroadcastTypeAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> BroadcastTypeAttributeRef<'c, 't> {
    /// Returns the [`BroadcastType`] value stored in this attribute.
    pub fn value(&self) -> BroadcastType {
        let rendered = self.to_string();
        let value = parse_enum_attribute_value(rendered, "broadcast")
            .and_then(|value| BroadcastType::try_from(value.as_str()).ok())
            .unwrap();
        value
    }
}

impl<'c, 't> Attribute<'c, 't> for BroadcastTypeAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        let gpu_attribute = attribute.cast::<GpuAttributeRef>()?;
        if gpu_attribute.mnemonic().as_deref() == Some("broadcast") { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(BroadcastTypeAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// MLIR [`Attribute`] wrapper for the `gpu.object` attribute.
#[derive(Copy, Clone)]
pub struct ObjectAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> ObjectAttributeRef<'c, 't> {
    /// Returns the target attribute stored in this object attribute.
    pub fn target(&self) -> AttributeRef<'c, 't> {
        unsafe { AttributeRef::from_c_api(mlirGPUObjectAttrGetTarget(self.handle), self.context).unwrap() }
    }

    /// Returns the object format identifier stored in this object attribute.
    pub fn format(&self) -> u32 {
        unsafe { mlirGPUObjectAttrGetFormat(self.handle) }
    }

    /// Returns the object payload stored in this object attribute.
    pub fn object(&self) -> StringRef<'c> {
        unsafe { StringRef::from_c_api(mlirGPUObjectAttrGetObject(self.handle)) }
    }

    /// Returns the object properties attribute, if present.
    pub fn properties(&self) -> Option<AttributeRef<'c, 't>> {
        if unsafe { mlirGPUObjectAttrHasProperties(self.handle) } {
            unsafe { AttributeRef::from_c_api(mlirGPUObjectAttrGetProperties(self.handle), self.context) }
        } else {
            None
        }
    }

    /// Returns the kernels attribute, if present.
    pub fn kernels(&self) -> Option<AttributeRef<'c, 't>> {
        if unsafe { mlirGPUObjectAttrHasKernels(self.handle) } {
            unsafe { AttributeRef::from_c_api(mlirGPUObjectAttrGetKernels(self.handle), self.context) }
        } else {
            None
        }
    }
}

mlir_subtype_trait_impls!(ObjectAttributeRef<'c, 't> as Attribute, mlir_type = Attribute, mlir_subtype = GPUObjectAttr);

impl<'t> Context<'t> {
    /// Parses a GPU dialect attribute from the provided source string.
    pub fn parse_gpu_attribute<'c, S: AsRef<str>>(&'c self, source: S) -> Option<GpuAttributeRef<'c, 't>> {
        self.load_dialect(DialectHandle::gpu());
        self.parse_attribute(source.as_ref()).and_then(|attribute| attribute.cast::<GpuAttributeRef>())
    }

    /// Creates a new `gpu.address_space` attribute owned by this [`Context`].
    pub fn gpu_address_space_attribute<'c>(&'c self, address_space: AddressSpace) -> AddressSpaceAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        let rendered = format!("#gpu.address_space<{}>", address_space.as_str());
        self.parse_attribute(&rendered)
            .and_then(|attribute| attribute.cast::<AddressSpaceAttributeRef>())
            .unwrap()
    }

    /// Creates a new `gpu.dim` attribute owned by this [`Context`].
    pub fn gpu_dimension_attribute<'c>(&'c self, dimension: Dimension) -> DimensionAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        let rendered = format!("#gpu<dim {}>", dimension.as_str());
        self.parse_attribute(&rendered)
            .and_then(|attribute| attribute.cast::<DimensionAttributeRef>())
            .unwrap()
    }

    /// Creates a new `gpu.all_reduce_op` attribute owned by this [`Context`].
    pub fn gpu_all_reduce_operation_attribute<'c>(
        &'c self,
        operation: AllReduceOperation,
    ) -> AllReduceOperationAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        let rendered = format!("#gpu<all_reduce_op {}>", operation.as_str());
        self.parse_attribute(&rendered)
            .and_then(|attribute| attribute.cast::<AllReduceOperationAttributeRef>())
            .unwrap()
    }

    /// Creates a new `gpu.shuffle_mode` attribute owned by this [`Context`].
    pub fn gpu_shuffle_mode_attribute<'c>(&'c self, mode: ShuffleMode) -> ShuffleModeAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        let rendered = format!("#gpu<shuffle_mode {}>", mode.as_str());
        self.parse_attribute(&rendered)
            .and_then(|attribute| attribute.cast::<ShuffleModeAttributeRef>())
            .unwrap()
    }

    /// Creates a new `gpu.mma_element_wise` attribute owned by this [`Context`].
    pub fn gpu_mma_elementwise_attribute<'c>(&'c self, op: MmaElementwiseOp) -> MmaElementwiseOpAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        let rendered = format!("#gpu<mma_element_wise {}>", op.as_str());
        self.parse_attribute(&rendered)
            .and_then(|attribute| attribute.cast::<MmaElementwiseOpAttributeRef>())
            .unwrap()
    }

    /// Creates a new `gpu.prune_2to4_spmat_flag` attribute owned by this [`Context`].
    pub fn gpu_prune_2to4_spmat_flag_attribute<'c>(
        &'c self,
        flag: Prune2To4SpMatFlag,
    ) -> Prune2To4SpMatFlagAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        let rendered = format!("#gpu<prune_2to4_spmat_flag {}>", flag.as_str());
        self.parse_attribute(&rendered)
            .and_then(|attribute| attribute.cast::<Prune2To4SpMatFlagAttributeRef>())
            .unwrap()
    }

    /// Creates a new `gpu.mat_transpose_mode` attribute owned by this [`Context`].
    pub fn gpu_transpose_mode_attribute<'c>(&'c self, mode: TransposeMode) -> TransposeModeAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        let rendered = format!("#gpu<mat_transpose_mode {}>", mode.as_str());
        self.parse_attribute(&rendered)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap()
    }

    /// Creates a new `gpu.spgemm_work_estimation_or_compute_kind` attribute owned by this [`Context`].
    pub fn gpu_spgemm_work_estimation_or_compute_kind_attribute<'c>(
        &'c self,
        kind: SpGemmWorkEstimationOrComputeKind,
    ) -> SpGemmWorkEstimationOrComputeKindAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        let rendered = format!("#gpu<spgemm_work_estimation_or_compute_kind {}>", kind.as_str());
        self.parse_attribute(&rendered)
            .and_then(|attribute| attribute.cast::<SpGemmWorkEstimationOrComputeKindAttributeRef>())
            .unwrap()
    }

    /// Creates a new `gpu.broadcast` attribute owned by this [`Context`].
    pub fn gpu_broadcast_type_attribute<'c>(
        &'c self,
        broadcast_type: BroadcastType,
    ) -> BroadcastTypeAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        let rendered = format!("#gpu<broadcast {}>", broadcast_type.as_str());
        self.parse_attribute(&rendered)
            .and_then(|attribute| attribute.cast::<BroadcastTypeAttributeRef>())
            .unwrap()
    }

    /// Creates a new `gpu.object` attribute owned by this [`Context`].
    pub fn gpu_object_attribute<'c, S: AsRef<str>, A: Attribute<'c, 't>>(
        &'c self,
        target: A,
        format: u32,
        object: S,
        properties: Option<AttributeRef<'c, 't>>,
    ) -> ObjectAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        unsafe {
            let properties = properties.unwrap_or_else(|| self.null_attribute());
            ObjectAttributeRef::from_c_api(
                mlirGPUObjectAttrGet(
                    *self.handle.borrow(),
                    target.to_c_api(),
                    format,
                    StringRef::from(object.as_ref()).to_c_api(),
                    properties.to_c_api(),
                ),
                &self,
            )
            .unwrap()
        }
    }

    /// Creates a new `gpu.object` attribute with an explicit kernels attribute.
    pub fn gpu_object_attribute_with_kernels<'c, S: AsRef<str>, A: Attribute<'c, 't>>(
        &'c self,
        target: A,
        format: u32,
        object: S,
        properties: Option<AttributeRef<'c, 't>>,
        kernels: Option<AttributeRef<'c, 't>>,
    ) -> ObjectAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        unsafe {
            let properties = properties.unwrap_or_else(|| self.null_attribute());
            let kernels = kernels.unwrap_or_else(|| self.null_attribute());
            ObjectAttributeRef::from_c_api(
                mlirGPUObjectAttrGetWithKernels(
                    *self.handle.borrow(),
                    target.to_c_api(),
                    format,
                    StringRef::from(object.as_ref()).to_c_api(),
                    properties.to_c_api(),
                    kernels.to_c_api(),
                ),
                &self,
            )
            .unwrap()
        }
    }
}

fn attribute_is_gpu<'c, 't: 'c, A: Attribute<'c, 't>>(attribute: A) -> bool {
    attribute.dialect().namespace().ok().map(|namespace| namespace == "gpu").unwrap_or(false)
        || attribute
            .cast::<OpaqueAttributeRef>()
            .and_then(|opaque_attribute| opaque_attribute.dialect_namespace().ok().map(|namespace| namespace == "gpu"))
            .unwrap_or(false)
}

fn attribute_mnemonic<'c, 't: 'c, A: Attribute<'c, 't>>(attribute: A) -> Option<String> {
    let rendered = attribute.to_string();
    if let Some(suffix) = rendered.strip_prefix("#gpu.") {
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
        return Some(suffix[..end].to_owned());
    }

    let contents = rendered.strip_prefix("#gpu<")?.strip_suffix('>')?;
    contents.split_whitespace().next().map(ToOwned::to_owned)
}

fn parse_enum_attribute_value(rendered: String, mnemonic: &str) -> Option<String> {
    let rendered = rendered.trim().to_owned();
    let legacy_prefix = format!("#gpu.{mnemonic}<");
    if let Some(value) = rendered.strip_prefix(&legacy_prefix).and_then(|suffix| suffix.strip_suffix('>')) {
        return Some(value.to_owned());
    }

    let contents = rendered.strip_prefix("#gpu<")?.strip_suffix('>')?;
    let mut tokens = contents.split_whitespace();
    if tokens.next()? != mnemonic {
        return None;
    }
    let value = tokens.collect::<Vec<_>>().join(" ");
    if value.is_empty() { None } else { Some(value) }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_gpu_attribute_from_opaque_attribute() {
        let context = Context::new();

        let attribute = context.opaque_attribute("gpu", "address_space<global>", context.none_type());
        let attribute = attribute.cast::<GpuAttributeRef>();
        assert!(attribute.is_some());

        let not_gpu = context.opaque_attribute("nvvm", "target<chip = \"sm_70\">", context.none_type());
        assert!(not_gpu.cast::<GpuAttributeRef>().is_none());
    }

    #[test]
    fn test_parse_gpu_attribute() {
        let context = Context::new();
        let parsed = context.parse_gpu_attribute("#gpu.address_space<global>");
        assert!(parsed.is_some());
        assert_eq!(parsed.unwrap().mnemonic().as_deref(), Some("address_space"));
    }

    #[test]
    fn test_gpu_address_space_attribute() {
        let context = Context::new();
        let attribute = context.gpu_address_space_attribute(AddressSpace::Workgroup);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), AddressSpace::Workgroup);
        test_attribute_display_and_debug(attribute, "#gpu.address_space<workgroup>");
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_gpu_object_attribute() {
        let context = Context::new();
        let target = context.opaque_attribute("nvvm", "target<chip = \"sm_70\">", context.none_type());
        let attribute = context.gpu_object_attribute(target, 0, "binary", None);

        assert_eq!(attribute.target(), target);
        assert_eq!(attribute.format(), 0);
        assert_eq!(attribute.object().as_str().unwrap(), "binary");
        assert_eq!(attribute.properties(), None);
        assert_eq!(attribute.kernels(), None);
    }

    #[test]
    fn test_gpu_object_attribute_with_kernels() {
        let context = Context::new();
        let target = context.opaque_attribute("nvvm", "target<chip = \"sm_80\">", context.none_type());
        let kernels = context.array_attribute(&[context.string_attribute("kernel0")]);
        let attribute =
            context.gpu_object_attribute_with_kernels(target, 1, "cubin", None, Some(kernels.as_ref()));

        assert_eq!(attribute.target(), target);
        assert_eq!(attribute.format(), 1);
        assert_eq!(attribute.object().as_str().unwrap(), "cubin");
        assert_eq!(attribute.properties(), None);
        assert_eq!(attribute.kernels().unwrap(), kernels);
    }

    #[test]
    fn test_gpu_enum_attributes() {
        let context = Context::new();

        let dimension = context.gpu_dimension_attribute(Dimension::Y);
        assert_eq!(dimension.value(), Dimension::Y);
        test_attribute_display_and_debug(dimension, "#gpu<dim y>");

        let reduce = context.gpu_all_reduce_operation_attribute(AllReduceOperation::MaxSi);
        assert_eq!(reduce.value(), AllReduceOperation::MaxSi);
        test_attribute_display_and_debug(reduce, "#gpu<all_reduce_op maxsi>");

        let shuffle = context.gpu_shuffle_mode_attribute(ShuffleMode::Idx);
        assert_eq!(shuffle.value(), ShuffleMode::Idx);
        test_attribute_display_and_debug(shuffle, "#gpu<shuffle_mode idx>");

        let elementwise = context.gpu_mma_elementwise_attribute(MmaElementwiseOp::DivU);
        assert_eq!(elementwise.value(), MmaElementwiseOp::DivU);
        test_attribute_display_and_debug(elementwise, "#gpu<mma_element_wise divu>");

        let prune = context.gpu_prune_2to4_spmat_flag_attribute(Prune2To4SpMatFlag::PruneAndCheck);
        assert_eq!(prune.value(), Prune2To4SpMatFlag::PruneAndCheck);
        test_attribute_display_and_debug(prune, "#gpu<prune_2to4_spmat_flag PRUNE_AND_CHECK>");

        let transpose = context.gpu_transpose_mode_attribute(TransposeMode::Transpose);
        assert_eq!(transpose.value(), TransposeMode::Transpose);
        test_attribute_display_and_debug(transpose, "#gpu<mat_transpose_mode TRANSPOSE>");

        let kind = context
            .gpu_spgemm_work_estimation_or_compute_kind_attribute(SpGemmWorkEstimationOrComputeKind::WorkEstimation);
        assert_eq!(kind.value(), SpGemmWorkEstimationOrComputeKind::WorkEstimation);
        test_attribute_display_and_debug(kind, "#gpu<spgemm_work_estimation_or_compute_kind WORK_ESTIMATION>");

        let broadcast = context.gpu_broadcast_type_attribute(BroadcastType::FirstActiveLane);
        assert_eq!(broadcast.value(), BroadcastType::FirstActiveLane);
        test_attribute_display_and_debug(broadcast, "#gpu<broadcast first_active_lane>");
    }
}
