use ryft_xla_sys::bindings::{
    MlirAttribute, mlirAttributeIsAGPUObjectAttr, mlirGPUObjectAttrGet, mlirGPUObjectAttrGetFormat,
    mlirGPUObjectAttrGetKernels, mlirGPUObjectAttrGetObject, mlirGPUObjectAttrGetProperties,
    mlirGPUObjectAttrGetTarget, mlirGPUObjectAttrGetWithKernels, mlirGPUObjectAttrHasKernels,
    mlirGPUObjectAttrHasProperties,
};

use crate::{
    ArrayAttributeRef, Attribute, AttributeRef, Context, DialectHandle, DictionaryAttributeRef, FromWithContext,
    FunctionTypeRef, StringRef, mlir_subtype_trait_impls,
};

/// GPU compilation object format.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ObjectFormat {
    /// Generic target-dependent offload object.
    Offload = 1,

    /// GPU assembly object.
    Assembly = 2,

    /// Single-architecture GPU executable object.
    Binary = 3,

    /// Multi-architecture GPU fat binary object.
    FatBinary = 4,
}

impl ObjectFormat {
    /// Returns the MLIR spelling used in `#gpu.object` attributes.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Offload => "offload",
            Self::Assembly => "assembly",
            Self::Binary => "bin",
            Self::FatBinary => "fatbin",
        }
    }

    /// Constructs an [`ObjectFormat`] from the integer value used by the MLIR C API.
    pub fn from_c_api(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::Offload),
            2 => Some(Self::Assembly),
            3 => Some(Self::Binary),
            4 => Some(Self::FatBinary),
            _ => None,
        }
    }
}

/// GPU object [`Attribute`] containing a target, a serialized object payload, optional properties, and optional kernel
/// metadata. The target attribute is expected by MLIR to implement or promise the GPU target attribute interface. The
/// C API exposes the object attribute directly, so this wrapper preserves the raw MLIR behavior and leaves semantic
/// validation to MLIR verification.
///
/// Refer to the [official MLIR GPU dialect documentation](https://mlir.llvm.org/docs/Dialects/GPU/#gpuobjectattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct ObjectAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> ObjectAttributeRef<'c, 't> {
    /// Returns the GPU target attribute.
    pub fn target(&self) -> AttributeRef<'c, 't> {
        unsafe { AttributeRef::from_c_api(mlirGPUObjectAttrGetTarget(self.handle), self.context).unwrap() }
    }

    /// Returns the object format.
    pub fn format(&self) -> ObjectFormat {
        ObjectFormat::from_c_api(unsafe { mlirGPUObjectAttrGetFormat(self.handle) }).expect("invalid GPU object format")
    }

    /// Returns the serialized object payload.
    pub fn object(&self) -> StringRef<'c> {
        unsafe { StringRef::from_c_api(mlirGPUObjectAttrGetObject(self.handle)) }
    }

    /// Returns the optional properties dictionary.
    pub fn properties(&self) -> Option<AttributeRef<'c, 't>> {
        if unsafe { mlirGPUObjectAttrHasProperties(self.handle) } {
            unsafe { AttributeRef::from_c_api(mlirGPUObjectAttrGetProperties(self.handle), self.context) }
        } else {
            None
        }
    }

    /// Returns the optional kernel table attribute.
    pub fn kernels(&self) -> Option<AttributeRef<'c, 't>> {
        if unsafe { mlirGPUObjectAttrHasKernels(self.handle) } {
            unsafe { AttributeRef::from_c_api(mlirGPUObjectAttrGetKernels(self.handle), self.context) }
        } else {
            None
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for ObjectAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsAGPUObjectAttr(handle) } {
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

mlir_subtype_trait_impls!(ObjectAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

fn gpu_attribute_has_mnemonic(source: &str, mnemonic: &str) -> bool {
    source.starts_with(format!("#gpu.{mnemonic}<").as_str())
        || source == format!("#gpu.{mnemonic}")
        || source.starts_with(format!("#gpu<{mnemonic} ").as_str())
}

/// GPU [`Attribute`] for storing metadata related to a compiled kernel. The current MLIR C API does not expose direct
/// field accessors for this attribute, so this wrapper is specialized by checking the printed dialect attribute
/// spelling and constructed through the MLIR parser.
///
/// Refer to the [official MLIR GPU dialect documentation](https://mlir.llvm.org/docs/Dialects/GPU/#gpukernelmetadataattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct KernelMetadataAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for KernelMetadataAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        if gpu_attribute_has_mnemonic(attribute.to_string().as_str(), "kernel_metadata") {
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

mlir_subtype_trait_impls!(KernelMetadataAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// GPU [`Attribute`] representing a table of [`KernelMetadataAttributeRef`] values.
///
/// The current MLIR C API does not expose direct field accessors for this attribute, so this wrapper is specialized by
/// checking the printed dialect attribute spelling and constructed through the MLIR parser.
///
/// Refer to the [official MLIR GPU dialect documentation](https://mlir.llvm.org/docs/Dialects/GPU/#gpukerneltableattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct KernelTableAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for KernelTableAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        if gpu_attribute_has_mnemonic(attribute.to_string().as_str(), "kernel_table") {
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

mlir_subtype_trait_impls!(KernelTableAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// GPU offloading handler [`Attribute`] that selects one GPU object for embedding.
///
/// Refer to the [official MLIR GPU dialect documentation](https://mlir.llvm.org/docs/Dialects/GPU/#gpuselectobjectattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct SelectObjectAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for SelectObjectAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        if gpu_attribute_has_mnemonic(attribute.to_string().as_str(), "select_object") {
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

mlir_subtype_trait_impls!(SelectObjectAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Extracts the enum value payload from a textual GPU enum attribute if it uses the expected mnemonic.
fn gpu_enum_attribute_value(source: &str, mnemonic: &str) -> Option<String> {
    let prefixed_prefix = format!("#gpu.{mnemonic}<");
    let dialect_prefix = format!("#gpu<{mnemonic} ");
    let plain_prefix = format!("#gpu.{mnemonic} ");
    source
        .strip_prefix(prefixed_prefix.as_str())
        .and_then(|source| source.strip_suffix(">"))
        .or_else(|| source.strip_prefix(dialect_prefix.as_str()).and_then(|source| source.strip_suffix(">")))
        .or_else(|| source.strip_prefix(plain_prefix.as_str()))
        .map(|source| source.trim_matches('"'))
        .map(str::to_owned)
}

/// Builds the textual MLIR source for a GPU enum attribute using its mnemonic and MLIR-spelled value.
fn gpu_enum_attribute_source(mnemonic: &str, value: &str, uses_prefixed_format: bool) -> String {
    if uses_prefixed_format { format!("#gpu.{mnemonic}<\"{value}\">") } else { format!("#gpu<{mnemonic} \"{value}\">") }
}

macro_rules! gpu_enum_attribute {
    (
        enum_name = $enum_name:ident,
        attribute_name = $attribute_name:ident,
        context_method = $context_method:ident,
        mnemonic = $mnemonic:literal,
        uses_prefixed_format = $uses_prefixed_format:literal,
        description = $description:literal,
        variants = { $($variant:ident => $value:literal),+ $(,)* } $(,)*
    ) => {
        #[doc = "GPU "]
        #[doc = $description]
        #[doc = "."]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub enum $enum_name {
            $($variant,)+
        }

        impl $enum_name {
            /// Returns the MLIR spelling of this enum value.
            pub fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $value,)+
                }
            }

            /// Parses the MLIR spelling of this enum value.
            pub fn from_str(value: &str) -> Option<Self> {
                match value {
                    $($value => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }

        #[doc = "GPU "]
        #[doc = $description]
        #[doc = " [`Attribute`]."]
        #[derive(Copy, Clone)]
        pub struct $attribute_name<'c, 't> {
            /// Handle that represents this [`Attribute`] in the MLIR C API.
            handle: MlirAttribute,

            /// [`Context`] that owns this [`Attribute`].
            context: &'c Context<'t>,
        }

        impl<'c, 't> $attribute_name<'c, 't> {
            /// Returns the enum value stored in this attribute.
            pub fn value(&self) -> $enum_name {
                gpu_enum_attribute_value(self.to_string().as_str(), $mnemonic)
                    .and_then(|value| $enum_name::from_str(value.as_str()))
                    .expect(concat!("invalid `#gpu.", $mnemonic, "` attribute"))
            }
        }

        impl<'c, 't> Attribute<'c, 't> for $attribute_name<'c, 't> {
            unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
                if handle.ptr.is_null() {
                    return None;
                }
                let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
                if gpu_enum_attribute_value(attribute.to_string().as_str(), $mnemonic)
                    .and_then(|value| $enum_name::from_str(value.as_str()))
                    .is_some()
                {
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

        mlir_subtype_trait_impls!($attribute_name<'c, 't> as Attribute, mlir_type = Attribute);

        impl<'c, 't> FromWithContext<'c, 't, $enum_name> for $attribute_name<'c, 't> {
            fn from_with_context(value: $enum_name, context: &'c Context<'t>) -> Self {
                context.$context_method(value)
            }
        }

        impl<'t> Context<'t> {
            #[doc = "Creates a GPU "]
            #[doc = $description]
            #[doc = " attribute owned by this [`Context`]."]
            pub fn $context_method<'c>(&'c self, value: $enum_name) -> $attribute_name<'c, 't> {
                self.load_dialect(DialectHandle::gpu());
                self.parse_attribute(
                    gpu_enum_attribute_source($mnemonic, value.as_str(), $uses_prefixed_format).as_str(),
                )
                    .and_then(|attribute| attribute.cast())
                    .expect(concat!("invalid arguments to `Context::", stringify!($context_method), "`"))
            }
        }
    };
}

gpu_enum_attribute!(
    enum_name = AddressSpace,
    attribute_name = AddressSpaceAttributeRef,
    context_method = gpu_address_space_attribute,
    mnemonic = "address_space",
    uses_prefixed_format = true,
    description = "address space",
    variants = {
        Global => "global",
        Workgroup => "workgroup",
        Private => "private",
        Constant => "constant",
    },
);

gpu_enum_attribute!(
    enum_name = Dimension,
    attribute_name = DimensionAttributeRef,
    context_method = gpu_dimension_attribute,
    mnemonic = "dim",
    uses_prefixed_format = false,
    description = "dimension",
    variants = {
        X => "x",
        Y => "y",
        Z => "z",
    },
);

gpu_enum_attribute!(
    enum_name = AllReduceOperationKind,
    attribute_name = AllReduceOperationKindAttributeRef,
    context_method = gpu_all_reduce_operation_kind_attribute,
    mnemonic = "all_reduce_op",
    uses_prefixed_format = false,
    description = "all-reduce operation kind",
    variants = {
        Add => "add",
        Multiply => "mul",
        MinimumUnsignedInteger => "minui",
        MinimumSignedInteger => "minsi",
        MinimumNumberFloat => "minnumf",
        MaximumUnsignedInteger => "maxui",
        MaximumSignedInteger => "maxsi",
        MaximumNumberFloat => "maxnumf",
        And => "and",
        Or => "or",
        Xor => "xor",
        MinimumFloat => "minimumf",
        MaximumFloat => "maximumf",
    },
);

gpu_enum_attribute!(
    enum_name = ShuffleMode,
    attribute_name = ShuffleModeAttributeRef,
    context_method = gpu_shuffle_mode_attribute,
    mnemonic = "shuffle_mode",
    uses_prefixed_format = false,
    description = "shuffle mode",
    variants = {
        Xor => "xor",
        Down => "down",
        Up => "up",
        Index => "idx",
    },
);

gpu_enum_attribute!(
    enum_name = MmaElementwiseOperation,
    attribute_name = MmaElementwiseOperationAttributeRef,
    context_method = gpu_mma_elementwise_operation_attribute,
    mnemonic = "mma_element_wise",
    uses_prefixed_format = false,
    description = "MMA elementwise operation",
    variants = {
        AddFloat => "addf",
        MultiplyFloat => "mulf",
        SubtractFloat => "subf",
        MaximumFloat => "maxf",
        MinimumFloat => "minf",
        DivideFloat => "divf",
        AddInteger => "addi",
        MultiplyInteger => "muli",
        SubtractInteger => "subi",
        DivideSignedInteger => "divs",
        DivideUnsignedInteger => "divu",
        NegateFloat => "negatef",
        NegateSignedInteger => "negates",
        ExtendFloat => "extf",
        TruncateFloat => "truncf",
    },
);

gpu_enum_attribute!(
    enum_name = Prune2To4SparseMatrixFlag,
    attribute_name = Prune2To4SparseMatrixFlagAttributeRef,
    context_method = gpu_prune_2_to_4_sparse_matrix_flag_attribute,
    mnemonic = "prune_2to4_spmat_flag",
    uses_prefixed_format = false,
    description = "2-to-4 sparse matrix pruning flag",
    variants = {
        None => "NONE",
        PruneOnly => "PRUNE_ONLY",
        PruneAndCheck => "PRUNE_AND_CHECK",
    },
);

gpu_enum_attribute!(
    enum_name = MatrixTransposeMode,
    attribute_name = MatrixTransposeModeAttributeRef,
    context_method = gpu_matrix_transpose_mode_attribute,
    mnemonic = "mat_transpose_mode",
    uses_prefixed_format = false,
    description = "matrix transpose mode",
    variants = {
        NonTranspose => "NON_TRANSPOSE",
        Transpose => "TRANSPOSE",
        ConjugateTranspose => "CONJUGATE_TRANSPOSE",
    },
);

gpu_enum_attribute!(
    enum_name = SpGemmWorkKind,
    attribute_name = SpGemmWorkKindAttributeRef,
    context_method = gpu_sp_gemm_work_kind_attribute,
    mnemonic = "spgemm_work_estimation_or_compute_kind",
    uses_prefixed_format = false,
    description = "SpGEMM work kind",
    variants = {
        WorkEstimation => "WORK_ESTIMATION",
        Compute => "COMPUTE",
    },
);

gpu_enum_attribute!(
    enum_name = BroadcastType,
    attribute_name = BroadcastTypeAttributeRef,
    context_method = gpu_broadcast_type_attribute,
    mnemonic = "broadcast",
    uses_prefixed_format = false,
    description = "subgroup broadcast type",
    variants = {
        FirstActiveLane => "first_active_lane",
        SpecificLane => "specific_lane",
    },
);

impl<'t> Context<'t> {
    /// Creates a GPU [`KernelMetadataAttributeRef`] owned by this [`Context`].
    ///
    /// # Parameters
    ///
    ///   - `name`: Kernel symbol name.
    ///   - `function_type`: Function type used by the compiled kernel.
    ///   - `argument_attributes`: Optional argument attribute array following MLIR function interface constraints.
    ///   - `metadata`: Optional dictionary with target-specific kernel metadata.
    pub fn gpu_kernel_metadata_attribute<'c>(
        &'c self,
        name: &str,
        function_type: FunctionTypeRef<'c, 't>,
        argument_attributes: Option<ArrayAttributeRef<'c, 't>>,
        metadata: Option<DictionaryAttributeRef<'c, 't>>,
    ) -> KernelMetadataAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        let mut source = format!("#gpu.kernel_metadata<{}, {}", self.string_attribute(name), function_type);
        match (argument_attributes, metadata) {
            (Some(argument_attributes), Some(metadata)) => {
                source.push_str(format!(", arg_attrs = {argument_attributes}, metadata = {metadata}").as_str());
            }
            (Some(argument_attributes), None) => {
                source.push_str(format!(", arg_attrs = {argument_attributes}").as_str());
            }
            (None, Some(metadata)) => {
                source.push_str(format!(", metadata = {metadata}").as_str());
            }
            (None, None) => {}
        }
        source.push('>');
        self.parse_attribute(source.as_str())
            .and_then(|attribute| attribute.cast())
            .expect("invalid arguments to `Context::gpu_kernel_metadata_attribute`")
    }

    /// Creates a GPU [`KernelTableAttributeRef`] owned by this [`Context`].
    ///
    /// # Parameters
    ///
    ///   - `kernels`: Kernel metadata entries to include in the table.
    pub fn gpu_kernel_table_attribute<'c>(
        &'c self,
        kernels: &[KernelMetadataAttributeRef<'c, 't>],
    ) -> KernelTableAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        let source = if kernels.is_empty() {
            "#gpu.kernel_table<>".to_owned()
        } else {
            let kernels = kernels.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
            format!("#gpu.kernel_table<[{kernels}]>")
        };
        self.parse_attribute(source.as_str())
            .and_then(|attribute| attribute.cast())
            .expect("invalid arguments to `Context::gpu_kernel_table_attribute`")
    }

    /// Creates a GPU [`SelectObjectAttributeRef`] owned by this [`Context`].
    ///
    /// # Parameters
    ///
    ///   - `target`: Optional target selector. When omitted, MLIR selects the first object in a `gpu.binary`.
    pub fn gpu_select_object_attribute<'c>(
        &'c self,
        target: Option<AttributeRef<'c, 't>>,
    ) -> SelectObjectAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        let source =
            target.map_or_else(|| "#gpu.select_object".to_owned(), |target| format!("#gpu.select_object<{target}>"));
        self.parse_attribute(source.as_str())
            .and_then(|attribute| attribute.cast())
            .expect("invalid arguments to `Context::gpu_select_object_attribute`")
    }

    /// Creates a GPU [`ObjectAttributeRef`] owned by this [`Context`].
    ///
    /// # Parameters
    ///
    ///   - `target`: Attribute describing the target this object was built for.
    ///   - `format`: Object payload format.
    ///   - `object`: Serialized object payload.
    ///   - `properties`: Optional object properties dictionary.
    ///   - `kernels`: Optional kernel metadata table.
    pub fn gpu_object_attribute<'c, T: Attribute<'c, 't>, S: AsRef<str>>(
        &'c self,
        target: T,
        format: ObjectFormat,
        object: S,
        properties: Option<AttributeRef<'c, 't>>,
        kernels: Option<AttributeRef<'c, 't>>,
    ) -> ObjectAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::gpu());
        unsafe {
            let properties = properties.unwrap_or_else(|| self.null_attribute());
            let kernels = kernels.unwrap_or_else(|| self.null_attribute());
            let object = StringRef::from(object.as_ref());
            let handle = if kernels.to_c_api().ptr.is_null() {
                mlirGPUObjectAttrGet(
                    *self.handle.borrow(),
                    target.to_c_api(),
                    format as u32,
                    object.to_c_api(),
                    properties.to_c_api(),
                )
            } else {
                mlirGPUObjectAttrGetWithKernels(
                    *self.handle.borrow(),
                    target.to_c_api(),
                    format as u32,
                    object.to_c_api(),
                    properties.to_c_api(),
                    kernels.to_c_api(),
                )
            };
            ObjectAttributeRef::from_c_api(handle, self).expect("invalid arguments to `Context::gpu_object_attribute`")
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_object_format() {
        assert_eq!(ObjectFormat::Offload.as_str(), "offload");
        assert_eq!(ObjectFormat::Assembly.as_str(), "assembly");
        assert_eq!(ObjectFormat::Binary.as_str(), "bin");
        assert_eq!(ObjectFormat::FatBinary.as_str(), "fatbin");
        assert_eq!(ObjectFormat::from_c_api(1), Some(ObjectFormat::Offload));
        assert_eq!(ObjectFormat::from_c_api(2), Some(ObjectFormat::Assembly));
        assert_eq!(ObjectFormat::from_c_api(3), Some(ObjectFormat::Binary));
        assert_eq!(ObjectFormat::from_c_api(4), Some(ObjectFormat::FatBinary));
        assert_eq!(ObjectFormat::from_c_api(0), None);
    }

    #[test]
    fn test_object_attribute() {
        let context = Context::new();
        let target = context.unit_attribute();
        let properties = context.dictionary_attribute(&[]);
        let attribute =
            context.gpu_object_attribute(target, ObjectFormat::FatBinary, "payload", Some(properties.as_ref()), None);

        assert_eq!(attribute.target(), target);
        assert_eq!(attribute.format(), ObjectFormat::FatBinary);
        assert_eq!(attribute.object().as_str(), Ok("payload"));
        assert_eq!(attribute.properties(), Some(properties.as_ref()));
        assert_eq!(attribute.kernels(), None);
    }

    #[test]
    fn test_object_attribute_equality() {
        let context = Context::new();
        let target = context.unit_attribute();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.gpu_object_attribute(target, ObjectFormat::FatBinary, "payload", None, None);
        let attribute_2 = context.gpu_object_attribute(target, ObjectFormat::FatBinary, "payload", None, None);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.gpu_object_attribute(target, ObjectFormat::Binary, "payload", None, None);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let target = context.unit_attribute();
        let attribute_2 = context.gpu_object_attribute(target, ObjectFormat::FatBinary, "payload", None, None);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_object_attribute_display_and_debug() {
        let context = Context::new();
        let target = context.unit_attribute();
        let attribute = context.gpu_object_attribute(target, ObjectFormat::FatBinary, "payload", None, None);
        test_attribute_display_and_debug(attribute, "#gpu.object<unit, \"payload\">");
    }

    #[test]
    fn test_object_attribute_casting() {
        let context = Context::new();
        let target = context.unit_attribute();
        let attribute = context.gpu_object_attribute(target, ObjectFormat::FatBinary, "payload", None, None);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_compilation_attributes() {
        let context = Context::new();
        let function_type = context.function_type::<crate::TypeRef, crate::TypeRef>(&[], &[]);
        let metadata = context.dictionary_attribute(&[]);

        let kernel_metadata = context.gpu_kernel_metadata_attribute("kernel", function_type, None, Some(metadata));
        test_attribute_display_and_debug(kernel_metadata, r#"#gpu.kernel_metadata<"kernel", () -> (), metadata = {}>"#);
        test_attribute_casting(kernel_metadata);

        let kernel_table = context.gpu_kernel_table_attribute(&[kernel_metadata]);
        test_attribute_display_and_debug(
            kernel_table,
            r#"#gpu.kernel_table<[#gpu.kernel_metadata<"kernel", () -> (), metadata = {}>]>"#,
        );
        test_attribute_casting(kernel_table);

        let empty_kernel_table = context.gpu_kernel_table_attribute(&[]);
        test_attribute_display_and_debug(empty_kernel_table, "#gpu.kernel_table<>");

        let select_object = context.gpu_select_object_attribute(None);
        test_attribute_display_and_debug(select_object, "#gpu.select_object");
        test_attribute_casting(select_object);
    }

    #[test]
    fn test_address_space_attribute() {
        let context = Context::new();
        let attribute = context.gpu_address_space_attribute(AddressSpace::Workgroup);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), AddressSpace::Workgroup);
    }

    #[test]
    fn test_address_space_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.gpu_address_space_attribute(AddressSpace::Workgroup);
        let attribute_2 = context.gpu_address_space_attribute(AddressSpace::Workgroup);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.gpu_address_space_attribute(AddressSpace::Private);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.gpu_address_space_attribute(AddressSpace::Workgroup);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_address_space_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.gpu_address_space_attribute(AddressSpace::Workgroup);
        test_attribute_display_and_debug(attribute, "#gpu.address_space<workgroup>");
    }

    #[test]
    fn test_address_space_attribute_casting() {
        let context = Context::new();
        let attribute = context.gpu_address_space_attribute(AddressSpace::Workgroup);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_dimension_attribute() {
        let context = Context::new();
        let attribute = context.gpu_dimension_attribute(Dimension::X);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), Dimension::X);
    }

    #[test]
    fn test_dimension_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.gpu_dimension_attribute(Dimension::X);
        let attribute_2 = context.gpu_dimension_attribute(Dimension::X);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.gpu_dimension_attribute(Dimension::Y);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.gpu_dimension_attribute(Dimension::X);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_dimension_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.gpu_dimension_attribute(Dimension::X);
        test_attribute_display_and_debug(attribute, "#gpu<dim x>");
    }

    #[test]
    fn test_dimension_attribute_casting() {
        let context = Context::new();
        let attribute = context.gpu_dimension_attribute(Dimension::X);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_all_reduce_operation_kind_attribute() {
        let context = Context::new();
        let attribute = context.gpu_all_reduce_operation_kind_attribute(AllReduceOperationKind::Add);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), AllReduceOperationKind::Add);
    }

    #[test]
    fn test_all_reduce_operation_kind_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.gpu_all_reduce_operation_kind_attribute(AllReduceOperationKind::Add);
        let attribute_2 = context.gpu_all_reduce_operation_kind_attribute(AllReduceOperationKind::Add);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.gpu_all_reduce_operation_kind_attribute(AllReduceOperationKind::Xor);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.gpu_all_reduce_operation_kind_attribute(AllReduceOperationKind::Add);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_all_reduce_operation_kind_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.gpu_all_reduce_operation_kind_attribute(AllReduceOperationKind::Add);
        test_attribute_display_and_debug(attribute, "#gpu<all_reduce_op add>");
    }

    #[test]
    fn test_all_reduce_operation_kind_attribute_casting() {
        let context = Context::new();
        let attribute = context.gpu_all_reduce_operation_kind_attribute(AllReduceOperationKind::Add);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_shuffle_mode_attribute() {
        let context = Context::new();
        let attribute = context.gpu_shuffle_mode_attribute(ShuffleMode::Xor);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), ShuffleMode::Xor);
    }

    #[test]
    fn test_shuffle_mode_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.gpu_shuffle_mode_attribute(ShuffleMode::Xor);
        let attribute_2 = context.gpu_shuffle_mode_attribute(ShuffleMode::Xor);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.gpu_shuffle_mode_attribute(ShuffleMode::Down);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.gpu_shuffle_mode_attribute(ShuffleMode::Xor);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_shuffle_mode_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.gpu_shuffle_mode_attribute(ShuffleMode::Xor);
        test_attribute_display_and_debug(attribute, "#gpu<shuffle_mode xor>");
    }

    #[test]
    fn test_shuffle_mode_attribute_casting() {
        let context = Context::new();
        let attribute = context.gpu_shuffle_mode_attribute(ShuffleMode::Xor);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_mma_elementwise_operation_attribute() {
        let context = Context::new();
        let attribute = context.gpu_mma_elementwise_operation_attribute(MmaElementwiseOperation::AddFloat);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), MmaElementwiseOperation::AddFloat);
    }

    #[test]
    fn test_mma_elementwise_operation_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.gpu_mma_elementwise_operation_attribute(MmaElementwiseOperation::AddFloat);
        let attribute_2 = context.gpu_mma_elementwise_operation_attribute(MmaElementwiseOperation::AddFloat);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.gpu_mma_elementwise_operation_attribute(MmaElementwiseOperation::MultiplyFloat);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.gpu_mma_elementwise_operation_attribute(MmaElementwiseOperation::AddFloat);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_mma_elementwise_operation_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.gpu_mma_elementwise_operation_attribute(MmaElementwiseOperation::AddFloat);
        test_attribute_display_and_debug(attribute, "#gpu<mma_element_wise addf>");
    }

    #[test]
    fn test_mma_elementwise_operation_attribute_casting() {
        let context = Context::new();
        let attribute = context.gpu_mma_elementwise_operation_attribute(MmaElementwiseOperation::AddFloat);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_prune_2_to_4_sparse_matrix_flag_attribute() {
        let context = Context::new();
        let attribute = context.gpu_prune_2_to_4_sparse_matrix_flag_attribute(Prune2To4SparseMatrixFlag::PruneOnly);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), Prune2To4SparseMatrixFlag::PruneOnly);
    }

    #[test]
    fn test_prune_2_to_4_sparse_matrix_flag_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.gpu_prune_2_to_4_sparse_matrix_flag_attribute(Prune2To4SparseMatrixFlag::PruneOnly);
        let attribute_2 = context.gpu_prune_2_to_4_sparse_matrix_flag_attribute(Prune2To4SparseMatrixFlag::PruneOnly);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 =
            context.gpu_prune_2_to_4_sparse_matrix_flag_attribute(Prune2To4SparseMatrixFlag::PruneAndCheck);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.gpu_prune_2_to_4_sparse_matrix_flag_attribute(Prune2To4SparseMatrixFlag::PruneOnly);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_prune_2_to_4_sparse_matrix_flag_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.gpu_prune_2_to_4_sparse_matrix_flag_attribute(Prune2To4SparseMatrixFlag::PruneOnly);
        test_attribute_display_and_debug(attribute, "#gpu<prune_2to4_spmat_flag PRUNE_ONLY>");
    }

    #[test]
    fn test_prune_2_to_4_sparse_matrix_flag_attribute_casting() {
        let context = Context::new();
        let attribute = context.gpu_prune_2_to_4_sparse_matrix_flag_attribute(Prune2To4SparseMatrixFlag::PruneOnly);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_matrix_transpose_mode_attribute() {
        let context = Context::new();
        let attribute = context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::NonTranspose);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), MatrixTransposeMode::NonTranspose);
    }

    #[test]
    fn test_matrix_transpose_mode_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::NonTranspose);
        let attribute_2 = context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::NonTranspose);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::Transpose);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::NonTranspose);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_matrix_transpose_mode_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::NonTranspose);
        test_attribute_display_and_debug(attribute, "#gpu<mat_transpose_mode NON_TRANSPOSE>");
    }

    #[test]
    fn test_matrix_transpose_mode_attribute_casting() {
        let context = Context::new();
        let attribute = context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::NonTranspose);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_sp_gemm_work_kind_attribute() {
        let context = Context::new();
        let attribute = context.gpu_sp_gemm_work_kind_attribute(SpGemmWorkKind::WorkEstimation);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), SpGemmWorkKind::WorkEstimation);
    }

    #[test]
    fn test_sp_gemm_work_kind_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.gpu_sp_gemm_work_kind_attribute(SpGemmWorkKind::WorkEstimation);
        let attribute_2 = context.gpu_sp_gemm_work_kind_attribute(SpGemmWorkKind::WorkEstimation);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.gpu_sp_gemm_work_kind_attribute(SpGemmWorkKind::Compute);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.gpu_sp_gemm_work_kind_attribute(SpGemmWorkKind::WorkEstimation);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_sp_gemm_work_kind_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.gpu_sp_gemm_work_kind_attribute(SpGemmWorkKind::WorkEstimation);
        test_attribute_display_and_debug(attribute, "#gpu<spgemm_work_estimation_or_compute_kind WORK_ESTIMATION>");
    }

    #[test]
    fn test_sp_gemm_work_kind_attribute_casting() {
        let context = Context::new();
        let attribute = context.gpu_sp_gemm_work_kind_attribute(SpGemmWorkKind::WorkEstimation);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_broadcast_type_attribute() {
        let context = Context::new();
        let attribute = context.gpu_broadcast_type_attribute(BroadcastType::FirstActiveLane);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), BroadcastType::FirstActiveLane);
    }

    #[test]
    fn test_broadcast_type_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.gpu_broadcast_type_attribute(BroadcastType::FirstActiveLane);
        let attribute_2 = context.gpu_broadcast_type_attribute(BroadcastType::FirstActiveLane);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.gpu_broadcast_type_attribute(BroadcastType::SpecificLane);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.gpu_broadcast_type_attribute(BroadcastType::FirstActiveLane);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_broadcast_type_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.gpu_broadcast_type_attribute(BroadcastType::FirstActiveLane);
        test_attribute_display_and_debug(attribute, "#gpu<broadcast first_active_lane>");
    }

    #[test]
    fn test_broadcast_type_attribute_casting() {
        let context = Context::new();
        let attribute = context.gpu_broadcast_type_attribute(BroadcastType::FirstActiveLane);
        test_attribute_casting(attribute);
    }
}
