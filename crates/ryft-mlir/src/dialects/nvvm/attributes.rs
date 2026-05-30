use ryft_xla_sys::bindings::MlirAttribute;

use crate::{
    ArrayAttributeRef, Attribute, AttributeRef, Context, DialectHandle, DictionaryAttributeRef, Error,
    mlir_subtype_trait_impls,
};

macro_rules! nvvm_enum_attribute {
    (
        enum_name = $enum_name:ident,
        attribute_name = $attribute_name:ident,
        context_method = $context_method:ident,
        mnemonic = $mnemonic:literal,
        source = $source:literal,
        description = $description:literal,
        variants = { $($variant:ident => ($value:literal, $spelling:literal)),+ $(,)* },
    ) => {
        #[doc = "Represents an NVVM "]
        #[doc = $description]
        #[doc = " value."]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum $enum_name {
            $(
                #[doc = concat!("The `", $spelling, "` enum value.")]
                $variant,
            )+
        }

        impl $enum_name {
            /// All values in this NVVM enum.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            /// Returns the integer representation used by MLIR for this enum value.
            pub fn value(&self) -> u32 {
                match self {
                    $(Self::$variant => $value,)+
                }
            }

            /// Creates this enum from the integer representation used by MLIR.
            pub fn from_value(value: u32) -> Option<Self> {
                match value {
                    $($value => Some(Self::$variant),)+
                    _ => None,
                }
            }

            /// Returns the textual MLIR spelling for this enum value.
            pub fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $spelling,)+
                }
            }
        }

        #[doc = "MLIR [`Attribute`] that stores an NVVM "]
        #[doc = $description]
        #[doc = "."]
        #[derive(Copy, Clone)]
        pub struct $attribute_name<'c, 't> {
            /// Handle that represents this [`Attribute`] in the MLIR C API.
            handle: MlirAttribute,

            /// [`Context`] that owns this [`Attribute`].
            context: &'c Context<'t>,
        }

        impl $attribute_name<'_, '_> {
            /// Returns the enum value stored in this attribute.
            pub fn value(&self) -> Result<$enum_name, Error> {
                for value in $enum_name::ALL.iter().copied() {
                    if *self == self.context.$context_method(value)? {
                        return Ok(value);
                    }
                }
                Err(Error::invalid_argument(concat!("invalid nvvm ", $description, " attribute")))
            }
        }

        impl<'c, 't> Attribute<'c, 't> for $attribute_name<'c, 't> {
            unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
                let attribute = unsafe { AttributeRef::from_c_api(handle, context)? };
                let source = attribute.to_string();
                if source.starts_with(concat!("#nvvm.", $mnemonic, "<"))
                    || source.starts_with(concat!("#nvvm<", $mnemonic, " "))
                {
                    Ok(Self { handle, context })
                } else {
                    Err(Error::invalid_argument(concat!("expected nvvm ", $description, " attribute")))
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

        impl<'t> Context<'t> {
            #[doc = "Creates an NVVM "]
            #[doc = $description]
            #[doc = " attribute owned by this [`Context`]."]
            pub fn $context_method<'c>(&'c self, value: $enum_name) -> Result<$attribute_name<'c, 't>, Error> {
                self.load_dialect(DialectHandle::nvvm()?)?;
                let source = format!($source, value.as_str());
                self.parse_attribute(&source)?.cast::<$attribute_name>().ok_or_else(|| {
                    Error::invalid_argument(concat!("invalid nvvm ", $description, " attribute"))
                })
            }
        }
    };
}

nvvm_enum_attribute!(
    enum_name = FloatingPointRoundingMode,
    attribute_name = FloatingPointRoundingModeAttributeRef,
    context_method = nvvm_floating_point_rounding_mode_attribute,
    mnemonic = "fp_rnd_mode",
    source = "#nvvm.fp_rnd_mode<{}>",
    description = "floating-point rounding mode",
    variants = {
        None => (0, "none"),
        NearestEven => (1, "rn"),
        Downward => (2, "rm"),
        Upward => (3, "rp"),
        TowardZero => (4, "rz"),
        NearestAway => (5, "rna"),
        Stochastic => (6, "rs"),
    },
);

nvvm_enum_attribute!(
    enum_name = SaturationMode,
    attribute_name = SaturationModeAttributeRef,
    context_method = nvvm_saturation_mode_attribute,
    mnemonic = "sat_mode",
    source = "#nvvm.sat_mode<{}>",
    description = "saturation mode",
    variants = {
        None => (0, "none"),
        Satfinite => (1, "satfinite"),
        Sat => (2, "sat"),
    },
);

nvvm_enum_attribute!(
    enum_name = CacheEvictionPriority,
    attribute_name = CacheEvictionPriorityAttributeRef,
    context_method = nvvm_cache_eviction_priority_attribute,
    mnemonic = "cache_eviction_priority",
    source = "#nvvm<cache_eviction_priority {}>",
    description = "nvvm cache eviction priority",
    variants = {
        EvictNormal => (0, "evict_normal"),
        EvictFirst => (1, "evict_first"),
        EvictLast => (2, "evict_last"),
        EvictUnchanged => (3, "evict_unchanged"),
        NoAllocate => (4, "no_allocate"),
    },
);

nvvm_enum_attribute!(
    enum_name = MemorySpace,
    attribute_name = MemorySpaceAttributeRef,
    context_method = nvvm_memory_space_attribute,
    mnemonic = "memory_space",
    source = "#nvvm.memory_space<{}>",
    description = "nvvm memory space",
    variants = {
        Generic => (0, "generic"),
        Global => (1, "global"),
        Shared => (3, "shared"),
        Constant => (4, "constant"),
        Local => (5, "local"),
        Tensor => (6, "tensor"),
        SharedCluster => (7, "shared_cluster"),
    },
);

nvvm_enum_attribute!(
    enum_name = MemScopeKind,
    attribute_name = MemScopeKindAttributeRef,
    context_method = nvvm_mem_scope_kind_attribute,
    mnemonic = "mem_scope",
    source = "#nvvm.mem_scope<{}>",
    description = "nvvm memory scope kind",
    variants = {
        Cta => (0, "cta"),
        Cluster => (1, "cluster"),
        Gpu => (2, "gpu"),
        Sys => (3, "sys"),
    },
);

nvvm_enum_attribute!(
    enum_name = SharedSpace,
    attribute_name = SharedSpaceAttributeRef,
    context_method = nvvm_shared_space_attribute,
    mnemonic = "shared_space",
    source = "#nvvm.shared_space<{}>",
    description = "shared memory space",
    variants = {
        Cta => (0, "cta"),
        Cluster => (1, "cluster"),
    },
);

nvvm_enum_attribute!(
    enum_name = MemOrderKind,
    attribute_name = MemOrderKindAttributeRef,
    context_method = nvvm_mem_order_kind_attribute,
    mnemonic = "mem_order",
    source = "#nvvm.mem_order<{}>",
    description = "nvvm memory ordering kind",
    variants = {
        Weak => (0, "weak"),
        Relaxed => (1, "relaxed"),
        Acquire => (2, "acquire"),
        Release => (3, "release"),
        AcqRel => (4, "acq_rel"),
        Sc => (5, "sc"),
        Mmio => (6, "mmio"),
        Volatile => (7, "volatile"),
    },
);

nvvm_enum_attribute!(
    enum_name = ReductionKind,
    attribute_name = ReductionKindAttributeRef,
    context_method = nvvm_reduction_kind_attribute,
    mnemonic = "reduction_kind",
    source = "#nvvm<reduction_kind {}>",
    description = "reduction kind",
    variants = {
        Add => (1, "add"),
        And => (2, "and"),
        Max => (3, "max"),
        Min => (4, "min"),
        Or => (5, "or"),
        Umax => (6, "umax"),
        Umin => (7, "umin"),
        Xor => (8, "xor"),
        Fmin => (9, "fmin"),
        Fmax => (10, "fmax"),
    },
);

nvvm_enum_attribute!(
    enum_name = BarrierReduction,
    attribute_name = BarrierReductionAttributeRef,
    context_method = nvvm_barrier_reduction_attribute,
    mnemonic = "reduction",
    source = "#nvvm.reduction<{}>",
    description = "nvvm barrier reduction operation",
    variants = {
        Popc => (0, "popc"),
        And => (1, "and"),
        Or => (2, "or"),
    },
);

nvvm_enum_attribute!(
    enum_name = ProxyKind,
    attribute_name = ProxyKindAttributeRef,
    context_method = nvvm_proxy_kind_attribute,
    mnemonic = "proxy_kind",
    source = "#nvvm.proxy_kind<{}>",
    description = "proxy kind",
    variants = {
        Alias => (0, "alias"),
        Async => (1, "async"),
        AsyncGlobal => (2, "async.global"),
        AsyncShared => (3, "async.shared"),
        Tensormap => (4, "tensormap"),
        Generic => (5, "generic"),
    },
);

nvvm_enum_attribute!(
    enum_name = SetMaxRegisterAction,
    attribute_name = SetMaxRegisterActionAttributeRef,
    context_method = nvvm_set_max_register_action_attribute,
    mnemonic = "action",
    source = "#nvvm<action {}>",
    description = "nvvm set max register action",
    variants = {
        Decrease => (1, "decrease"),
        Increase => (0, "increase"),
    },
);

nvvm_enum_attribute!(
    enum_name = ShflKind,
    attribute_name = ShflKindAttributeRef,
    context_method = nvvm_shfl_kind_attribute,
    mnemonic = "shfl_kind",
    source = "#nvvm<shfl_kind {}>",
    description = "nvvm shuffle kind",
    variants = {
        Bfly => (0, "bfly"),
        Up => (1, "up"),
        Down => (2, "down"),
        Idx => (3, "idx"),
    },
);

nvvm_enum_attribute!(
    enum_name = VoteSyncKind,
    attribute_name = VoteSyncKindAttributeRef,
    context_method = nvvm_vote_sync_kind_attribute,
    mnemonic = "vote_sync_kind",
    source = "#nvvm<vote_sync_kind {}>",
    description = "nvvm vote sync kind",
    variants = {
        Any => (0, "any"),
        All => (1, "all"),
        Ballot => (2, "ballot"),
        Uni => (3, "uni"),
    },
);

nvvm_enum_attribute!(
    enum_name = PermuteMode,
    attribute_name = PermuteModeAttributeRef,
    context_method = nvvm_permute_mode_attribute,
    mnemonic = "permute_mode",
    source = "#nvvm.permute_mode<{}>",
    description = "nvvm permute mode",
    variants = {
        Default => (0, "default"),
        F4e => (1, "f4e"),
        B4e => (2, "b4e"),
        Rc8 => (3, "rc8"),
        Ecl => (4, "ecl"),
        Ecr => (5, "ecr"),
        Rc16 => (6, "rc16"),
    },
);

nvvm_enum_attribute!(
    enum_name = LoadCacheModifierKind,
    attribute_name = LoadCacheModifierKindAttributeRef,
    context_method = nvvm_load_cache_modifier_kind_attribute,
    mnemonic = "load_cache_modifier",
    source = "#nvvm<load_cache_modifier {}>",
    description = "nvvm load cache modifier kind",
    variants = {
        Ca => (0, "ca"),
        Cg => (1, "cg"),
        Cs => (2, "cs"),
        Lu => (3, "lu"),
        Cv => (4, "cv"),
    },
);

nvvm_enum_attribute!(
    enum_name = MmaB1Operation,
    attribute_name = MmaB1OperationAttributeRef,
    context_method = nvvm_mma_b1_operation_attribute,
    mnemonic = "mma_b1op",
    source = "#nvvm.mma_b1op<{}>",
    description = "mma binary operations",
    variants = {
        None => (0, "none"),
        XorPopc => (1, "xor_popc"),
        AndPopc => (2, "and_popc"),
    },
);

nvvm_enum_attribute!(
    enum_name = MmaIntegerOverflow,
    attribute_name = MmaIntegerOverflowAttributeRef,
    context_method = nvvm_mma_integer_overflow_attribute,
    mnemonic = "mma_int_overflow",
    source = "#nvvm.mma_int_overflow<{}>",
    description = "mma overflow options",
    variants = {
        Satfinite => (1, "satfinite"),
        Wrapped => (0, "wrapped"),
    },
);

nvvm_enum_attribute!(
    enum_name = MmaKind,
    attribute_name = MmaKindAttributeRef,
    context_method = nvvm_mma_kind_attribute,
    mnemonic = "mma_kind",
    source = "#nvvm.mma_kind<{}>",
    description = "mma operation kind",
    variants = {
        F8f6f4 => (0, "f8f6f4"),
    },
);

nvvm_enum_attribute!(
    enum_name = MmaLayout,
    attribute_name = MmaLayoutAttributeRef,
    context_method = nvvm_mma_layout_attribute,
    mnemonic = "mma_layout",
    source = "#nvvm.mma_layout<{}>",
    description = "nvvm mma layout",
    variants = {
        Row => (0, "row"),
        Col => (1, "col"),
    },
);

nvvm_enum_attribute!(
    enum_name = MmaType,
    attribute_name = MmaTypeAttributeRef,
    context_method = nvvm_mma_type_attribute,
    mnemonic = "mma_type",
    source = "#nvvm.mma_type<{}>",
    description = "nvvm mma types",
    variants = {
        F16 => (0, "f16"),
        F32 => (1, "f32"),
        Tf32 => (2, "tf32"),
        Bf16 => (9, "bf16"),
        S8 => (4, "s8"),
        U8 => (3, "u8"),
        S32 => (5, "s32"),
        S4 => (8, "s4"),
        U4 => (7, "u4"),
        B1 => (6, "b1"),
        F64 => (10, "f64"),
        E4m3 => (11, "e4m3"),
        E5m2 => (12, "e5m2"),
        E3m2 => (13, "e3m2"),
        E2m3 => (14, "e2m3"),
        E2m1 => (15, "e2m1"),
    },
);

nvvm_enum_attribute!(
    enum_name = MmaFragment,
    attribute_name = MmaFragmentAttributeRef,
    context_method = nvvm_mma_fragment_attribute,
    mnemonic = "mma_frag",
    source = "#nvvm.mma_frag<{}>",
    description = "nvvm mma frag type",
    variants = {
        A => (0, "a"),
        B => (1, "b"),
        C => (2, "c"),
    },
);

nvvm_enum_attribute!(
    enum_name = LdStMatrixEltType,
    attribute_name = LdStMatrixEltTypeAttributeRef,
    context_method = nvvm_ld_st_matrix_elt_type_attribute,
    mnemonic = "ld_st_matrix_elt_type",
    source = "#nvvm.ld_st_matrix_elt_type<{}>",
    description = "element type for ldmatrix and stmatrix",
    variants = {
        B16 => (0, "b16"),
        B8 => (1, "b8"),
        B8x16B6x16P32 => (2, "b8x16.b6x16_p32"),
        B8x16B4x16P64 => (3, "b8x16.b4x16_p64"),
    },
);

nvvm_enum_attribute!(
    enum_name = ScaleVecSize,
    attribute_name = ScaleVecSizeAttributeRef,
    context_method = nvvm_scale_vec_size_attribute,
    mnemonic = "scale_vec_size",
    source = "#nvvm.scale_vec_size<{}>",
    description = "mma scale vector sizes",
    variants = {
        X1 => (0, "x1"),
        X2 => (1, "x2"),
        X4 => (2, "x4"),
    },
);

nvvm_enum_attribute!(
    enum_name = BlockScaleFormat,
    attribute_name = BlockScaleFormatAttributeRef,
    context_method = nvvm_block_scale_format_attribute,
    mnemonic = "block_scale_format",
    source = "#nvvm.block_scale_format<{}>",
    description = "mma block scale format",
    variants = {
        Ue8m0 => (0, "ue8m0"),
        Ue4m3 => (1, "ue4m3"),
    },
);

nvvm_enum_attribute!(
    enum_name = MmaBlockScaleKind,
    attribute_name = MmaBlockScaleKindAttributeRef,
    context_method = nvvm_mma_block_scale_kind_attribute,
    mnemonic = "block_scale_kind",
    source = "#nvvm.block_scale_kind<{}>",
    description = "block scale kind",
    variants = {
        Mxf8f6f4 => (0, "mxf8f6f4"),
        Mxf4 => (1, "mxf4"),
        Mxf4nvf4 => (2, "mxf4nvf4"),
    },
);

nvvm_enum_attribute!(
    enum_name = TmaLoadMode,
    attribute_name = TmaLoadModeAttributeRef,
    context_method = nvvm_tma_load_mode_attribute,
    mnemonic = "tma_load_mode",
    source = "#nvvm.tma_load_mode<{}>",
    description = "nvvm tma load mode",
    variants = {
        Tile => (0, "tile"),
        Im2col => (1, "im2col"),
        Im2colW => (2, "im2col_w"),
        Im2colW128 => (3, "im2col_w_128"),
        TileGather4 => (4, "tile_gather4"),
    },
);

nvvm_enum_attribute!(
    enum_name = TmaStoreMode,
    attribute_name = TmaStoreModeAttributeRef,
    context_method = nvvm_tma_store_mode_attribute,
    mnemonic = "tma_store_mode",
    source = "#nvvm.tma_store_mode<{}>",
    description = "nvvm tma store mode",
    variants = {
        Tile => (0, "tile"),
        Im2col => (1, "im2col"),
        TileScatter4 => (2, "tile_scatter4"),
    },
);

nvvm_enum_attribute!(
    enum_name = CtaGroupKind,
    attribute_name = CtaGroupKindAttributeRef,
    context_method = nvvm_cta_group_kind_attribute,
    mnemonic = "cta_group",
    source = "#nvvm.cta_group<{}>",
    description = "nvvm cta group kind",
    variants = {
        Cta1 => (0, "cta_1"),
        Cta2 => (1, "cta_2"),
    },
);

nvvm_enum_attribute!(
    enum_name = PrefetchCacheLevel,
    attribute_name = PrefetchCacheLevelAttributeRef,
    context_method = nvvm_prefetch_cache_level_attribute,
    mnemonic = "prefetch_cache_level",
    source = "#nvvm<prefetch_cache_level {}>",
    description = "nvvm prefetch cache level",
    variants = {
        L1 => (0, "L1"),
        L2 => (1, "L2"),
    },
);

nvvm_enum_attribute!(
    enum_name = TmaReduxKind,
    attribute_name = TmaReduxKindAttributeRef,
    context_method = nvvm_tma_redux_kind_attribute,
    mnemonic = "tma_redux_kind",
    source = "#nvvm.tma_redux_kind<{}>",
    description = "nvvm tma redux kind",
    variants = {
        Add => (0, "add"),
        Max => (2, "max"),
        Min => (1, "min"),
        Inc => (3, "inc"),
        Dec => (4, "dec"),
        And => (5, "and"),
        Or => (6, "or"),
        Xor => (7, "xor"),
    },
);

nvvm_enum_attribute!(
    enum_name = WgmmaScaleIn,
    attribute_name = WgmmaScaleInAttributeRef,
    context_method = nvvm_wgmma_scale_in_attribute,
    mnemonic = "wgmma_scale_in",
    source = "#nvvm.wgmma_scale_in<{}>",
    description = "wgmma overflow options",
    variants = {
        One => (1, "one"),
    },
);

nvvm_enum_attribute!(
    enum_name = WgmmaScaleOut,
    attribute_name = WgmmaScaleOutAttributeRef,
    context_method = nvvm_wgmma_scale_out_attribute,
    mnemonic = "wgmma_scale_out",
    source = "#nvvm.wgmma_scale_out<{}>",
    description = "wgmma input predicate",
    variants = {
        Zero => (0, "zero"),
        One => (1, "one"),
    },
);

nvvm_enum_attribute!(
    enum_name = WgmmaType,
    attribute_name = WgmmaTypeAttributeRef,
    context_method = nvvm_wgmma_type_attribute,
    mnemonic = "wgmma_type",
    source = "#nvvm.wgmma_type<{}>",
    description = "nvvm wgmma types",
    variants = {
        F16 => (0, "f16"),
        Tf32 => (1, "tf32"),
        U8 => (2, "u8"),
        S8 => (3, "s8"),
        B1 => (4, "b1"),
        Bf16 => (5, "bf16"),
        E4m3 => (6, "e4m3"),
        E5m2 => (7, "e5m2"),
        F32 => (8, "f32"),
        S32 => (9, "s32"),
    },
);

nvvm_enum_attribute!(
    enum_name = GridDependencyAction,
    attribute_name = GridDependencyActionAttributeRef,
    context_method = nvvm_grid_dependency_action_attribute,
    mnemonic = "grid_dep_action",
    source = "#nvvm<grid_dep_action {}>",
    description = "action kind for grid dependency control",
    variants = {
        Wait => (0, "wait"),
        LaunchDependents => (1, "launch_dependents"),
    },
);

nvvm_enum_attribute!(
    enum_name = MatchSyncKind,
    attribute_name = MatchSyncKindAttributeRef,
    context_method = nvvm_match_sync_kind_attribute,
    mnemonic = "match_sync_kind",
    source = "#nvvm<match_sync_kind {}>",
    description = "nvvm match sync kind",
    variants = {
        Any => (0, "any"),
        All => (1, "all"),
    },
);

nvvm_enum_attribute!(
    enum_name = Tcgen05FenceKind,
    attribute_name = Tcgen05FenceKindAttributeRef,
    context_method = nvvm_tcgen05_fence_kind_attribute,
    mnemonic = "tcgen05_fence",
    source = "#nvvm.tcgen05_fence<{}>",
    description = "nvvm tcgen05 fence kind",
    variants = {
        Before => (0, "before"),
        After => (1, "after"),
    },
);

nvvm_enum_attribute!(
    enum_name = Tcgen05WaitKind,
    attribute_name = Tcgen05WaitKindAttributeRef,
    context_method = nvvm_tcgen05_wait_kind_attribute,
    mnemonic = "tcgen05_wait",
    source = "#nvvm.tcgen05_wait<{}>",
    description = "nvvm tcgen05 wait kind",
    variants = {
        Load => (0, "load"),
        Store => (1, "store"),
    },
);

nvvm_enum_attribute!(
    enum_name = Tcgen05CpShape,
    attribute_name = Tcgen05CpShapeAttributeRef,
    context_method = nvvm_tcgen05_cp_shape_attribute,
    mnemonic = "tcgen05_cp_shape",
    source = "#nvvm.tcgen05_cp_shape<{}>",
    description = "tcgen05 cp shapes",
    variants = {
        Shape128x256b => (0, "shape_128x256b"),
        Shape4x256b => (1, "shape_4x256b"),
        Shape128x128b => (2, "shape_128x128b"),
        Shape64x128b => (3, "shape_64x128b"),
        Shape32x128b => (4, "shape_32x128b"),
    },
);

nvvm_enum_attribute!(
    enum_name = Tcgen05CpMulticast,
    attribute_name = Tcgen05CpMulticastAttributeRef,
    context_method = nvvm_tcgen05_cp_multicast_attribute,
    mnemonic = "tcgen05_cp_multicast",
    source = "#nvvm.tcgen05_cp_multicast<{}>",
    description = "tcgen05 cp multicast",
    variants = {
        None => (0, "none"),
        Warpx20213 => (1, "warpx2_02_13"),
        Warpx20123 => (2, "warpx2_01_23"),
        Warpx4 => (3, "warpx4"),
    },
);

nvvm_enum_attribute!(
    enum_name = Tcgen05CpSrcFormat,
    attribute_name = Tcgen05CpSrcFormatAttributeRef,
    context_method = nvvm_tcgen05_cp_src_format_attribute,
    mnemonic = "tcgen05_cp_src_fmt",
    source = "#nvvm.tcgen05_cp_src_fmt<{}>",
    description = "tcgen05 cp source format",
    variants = {
        B6x16P32 => (0, "b6x16_p32"),
        B4x16P64 => (1, "b4x16_p64"),
    },
);

nvvm_enum_attribute!(
    enum_name = Tcgen05LdStShape,
    attribute_name = Tcgen05LdStShapeAttributeRef,
    context_method = nvvm_tcgen05_ld_st_shape_attribute,
    mnemonic = "tcgen05_ldst_shape",
    source = "#nvvm.tcgen05_ldst_shape<{}>",
    description = "tcgen05 load/store shape",
    variants = {
        Shape16x64b => (0, "shape_16x64b"),
        Shape16x128b => (1, "shape_16x128b"),
        Shape16x256b => (2, "shape_16x256b"),
        Shape32x32b => (3, "shape_32x32b"),
        Shape16x32bx2 => (4, "shape_16x32bx2"),
    },
);

nvvm_enum_attribute!(
    enum_name = DotAccumulateType,
    attribute_name = DotAccumulateTypeAttributeRef,
    context_method = nvvm_dot_accumulate_type_attribute,
    mnemonic = "dot_accumulate_type",
    source = "#nvvm.dot_accumulate_type<{}>",
    description = "dot accumulate type",
    variants = {
        Signed => (1, "signed"),
        Unsigned => (0, "unsigned"),
    },
);

nvvm_enum_attribute!(
    enum_name = ClusterLaunchControlQueryType,
    attribute_name = ClusterLaunchControlQueryTypeAttributeRef,
    context_method = nvvm_cluster_launch_control_query_type_attribute,
    mnemonic = "cluster_launch_control_query_type",
    source = "#nvvm<cluster_launch_control_query_type {}>",
    description = "cluster launch control query type",
    variants = {
        IsCanceled => (0, "is_canceled"),
        GetFirstCtaIdX => (1, "get_first_cta_id_x"),
        GetFirstCtaIdY => (2, "get_first_cta_id_y"),
        GetFirstCtaIdZ => (3, "get_first_cta_id_z"),
    },
);

nvvm_enum_attribute!(
    enum_name = Tcgen05MmaKind,
    attribute_name = Tcgen05MmaKindAttributeRef,
    context_method = nvvm_tcgen05_mma_kind_attribute,
    mnemonic = "tcgen05_mma_kind",
    source = "#nvvm.tcgen05_mma_kind<{}>",
    description = "tcgen05 mma supported types",
    variants = {
        F16 => (0, "f16"),
        Tf32 => (1, "tf32"),
        F8f6f4 => (2, "f8f6f4"),
        I8 => (3, "i8"),
        Mxf8f6f4 => (4, "mxf8f6f4"),
        Mxf4 => (5, "mxf4"),
        Mxf4nvf4 => (6, "mxf4nvf4"),
    },
);

nvvm_enum_attribute!(
    enum_name = Tcgen05MmaCollectorOperation,
    attribute_name = Tcgen05MmaCollectorOperationAttributeRef,
    context_method = nvvm_tcgen05_mma_collector_operation_attribute,
    mnemonic = "tcgen05_mma_collectorop",
    source = "#nvvm.tcgen05_mma_collectorop<{}>",
    description = "tcgen05.mma collector buffer operation",
    variants = {
        Discard => (0, "discard"),
        Lastuse => (1, "lastuse"),
        Fill => (2, "fill"),
        Use => (3, "use"),
    },
);

nvvm_enum_attribute!(
    enum_name = Tcgen05MmaBlockScale,
    attribute_name = Tcgen05MmaBlockScaleAttributeRef,
    context_method = nvvm_tcgen05_mma_block_scale_attribute,
    mnemonic = "tcgen05_mma_block_scale",
    source = "#nvvm.tcgen05_mma_block_scale<{}>",
    description = "tcgen05.mma block scale attribute",
    variants = {
        Default => (0, "default"),
        Block16 => (1, "block16"),
        Block32 => (2, "block32"),
    },
);

nvvm_enum_attribute!(
    enum_name = Tcgen05MmaCollectorBBuffer,
    attribute_name = Tcgen05MmaCollectorBBufferAttributeRef,
    context_method = nvvm_tcgen05_mma_collector_b_buffer_attribute,
    mnemonic = "tcgen05_mma_collectorb",
    source = "#nvvm.tcgen05_mma_collectorb<{}>",
    description = "tcgen05 mma collector buffer b attribute",
    variants = {
        B0 => (0, "b0"),
        B1 => (1, "b1"),
        B2 => (2, "b2"),
        B3 => (3, "b3"),
    },
);

nvvm_enum_attribute!(
    enum_name = TensormapField,
    attribute_name = TensormapFieldAttributeRef,
    context_method = nvvm_tensormap_field_attribute,
    mnemonic = "tensormap_field",
    source = "#nvvm<tensormap_field {}>",
    description = "nvvm tensormap field kind",
    variants = {
        GlobalAddress => (0, "global_address"),
        Rank => (1, "rank"),
        BoxDim => (2, "box_dim"),
        GlobalDim => (3, "global_dim"),
        GlobalStride => (4, "global_stride"),
        ElementStride => (5, "element_stride"),
        Elemtype => (6, "elemtype"),
        InterleaveLayout => (7, "interleave_layout"),
        SwizzleMode => (8, "swizzle_mode"),
        SwizzleAtomicity => (9, "swizzle_atomicity"),
        FillMode => (10, "fill_mode"),
    },
);

nvvm_enum_attribute!(
    enum_name = TensormapElemtype,
    attribute_name = TensormapElemtypeAttributeRef,
    context_method = nvvm_tensormap_elemtype_attribute,
    mnemonic = "tensormap_elemtype",
    source = "#nvvm.tensormap_elemtype<{}>",
    description = "nvvm tensormap elemtype",
    variants = {
        U8 => (0, "u8"),
        U16 => (1, "u16"),
        U32 => (2, "u32"),
        S32 => (3, "s32"),
        U64 => (4, "u64"),
        S64 => (5, "s64"),
        F16 => (6, "f16"),
        F32 => (7, "f32"),
        F32Ftz => (8, "f32.ftz"),
        F64 => (9, "f64"),
        Bf16 => (10, "bf16"),
        Tf32 => (11, "tf32"),
        Tf32Ftz => (12, "tf32.ftz"),
        B4x16 => (13, "b4x16"),
        B4x16P64 => (14, "b4x16_p64"),
        B6x16P32 => (15, "b6x16_p32"),
    },
);

nvvm_enum_attribute!(
    enum_name = TensormapInterleaveLayout,
    attribute_name = TensormapInterleaveLayoutAttributeRef,
    context_method = nvvm_tensormap_interleave_layout_attribute,
    mnemonic = "tensormap_interleave_layout",
    source = "#nvvm.tensormap_interleave_layout<{}>",
    description = "nvvm tensormap interleave layout",
    variants = {
        NoInterleave => (0, "no_interleave"),
        B16 => (1, "b16"),
        B32 => (2, "b32"),
    },
);

nvvm_enum_attribute!(
    enum_name = TensormapSwizzleMode,
    attribute_name = TensormapSwizzleModeAttributeRef,
    context_method = nvvm_tensormap_swizzle_mode_attribute,
    mnemonic = "tensormap_swizzle_mode",
    source = "#nvvm.tensormap_swizzle_mode<{}>",
    description = "nvvm tensormap swizzle mode",
    variants = {
        NoSwizzling => (0, "no_swizzling"),
        B32 => (1, "b32"),
        B64 => (2, "b64"),
        B128 => (3, "b128"),
        B96 => (4, "b96"),
    },
);

nvvm_enum_attribute!(
    enum_name = TensormapSwizzleAtomicity,
    attribute_name = TensormapSwizzleAtomicityAttributeRef,
    context_method = nvvm_tensormap_swizzle_atomicity_attribute,
    mnemonic = "tensormap_swizzle_atomicity",
    source = "#nvvm.tensormap_swizzle_atomicity<{}>",
    description = "nvvm tensormap swizzle atomicity",
    variants = {
        B16 => (0, "b16"),
        B32 => (1, "b32"),
        B32FlipB8 => (2, "b32_flip_b8"),
        B64 => (3, "b64"),
    },
);

nvvm_enum_attribute!(
    enum_name = TensormapFillMode,
    attribute_name = TensormapFillModeAttributeRef,
    context_method = nvvm_tensormap_fill_mode_attribute,
    mnemonic = "tensormap_fill_mode",
    source = "#nvvm.tensormap_fill_mode<{}>",
    description = "nvvm tensormap fill mode",
    variants = {
        Zero => (0, "zero"),
        OobNan => (1, "oob_nan"),
    },
);

/// Shape parameters for NVVM MMA operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MmaShape {
    /// The `m` shape dimension.
    pub m: i32,

    /// The `n` shape dimension.
    pub n: i32,

    /// The `k` shape dimension.
    pub k: i32,
}

/// Shape parameters for NVVM `ldmatrix` and `stmatrix` operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LdStMatrixShape {
    /// The `m` shape dimension.
    pub m: i32,

    /// The `n` shape dimension.
    pub n: i32,
}

fn parse_integer_parameters<const N: usize>(source: &str, prefix: &str) -> Result<[i32; N], Error> {
    let body = source
        .strip_prefix(prefix)
        .and_then(|source| source.strip_prefix('<'))
        .and_then(|source| source.strip_suffix('>'))
        .ok_or_else(|| Error::invalid_argument(format!("invalid `{prefix}` attribute")))?;
    let values = body
        .split(',')
        .map(|part| part.split('=').next_back().unwrap_or(part).trim().parse::<i32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error::invalid_argument(format!("invalid `{prefix}` integer parameter")))?;
    values
        .try_into()
        .map_err(|_| Error::invalid_argument(format!("invalid `{prefix}` parameter count")))
}

fn escape_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

/// MLIR [`Attribute`] that stores an NVVM MMA shape.
#[derive(Copy, Clone)]
pub struct MmaShapeAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl MmaShapeAttributeRef<'_, '_> {
    /// Returns the NVVM MMA shape stored in this attribute.
    pub fn value(&self) -> Result<MmaShape, Error> {
        let values = parse_integer_parameters::<3>(&self.to_string(), "#nvvm.shape")?;
        Ok(MmaShape { m: values[0], n: values[1], k: values[2] })
    }
}

impl<'c, 't> Attribute<'c, 't> for MmaShapeAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context)? };
        if attribute.to_string().starts_with("#nvvm.shape") {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected nvvm MMA shape attribute"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(MmaShapeAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// MLIR [`Attribute`] that stores an NVVM ld/st matrix shape.
#[derive(Copy, Clone)]
pub struct LdStMatrixShapeAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl LdStMatrixShapeAttributeRef<'_, '_> {
    /// Returns the NVVM ld/st matrix shape stored in this attribute.
    pub fn value(&self) -> Result<LdStMatrixShape, Error> {
        let values = parse_integer_parameters::<2>(&self.to_string(), "#nvvm.ld_st_matrix_shape")?;
        Ok(LdStMatrixShape { m: values[0], n: values[1] })
    }
}

impl<'c, 't> Attribute<'c, 't> for LdStMatrixShapeAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context)? };
        if attribute.to_string().starts_with("#nvvm.ld_st_matrix_shape") {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected nvvm ld/st matrix shape attribute"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(LdStMatrixShapeAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// MLIR [`Attribute`] that stores an NVVM target configuration.
#[derive(Copy, Clone)]
pub struct TargetAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for TargetAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context)? };
        if attribute.to_string().starts_with("#nvvm.target") {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected nvvm target attribute"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(TargetAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates an NVVM MMA shape attribute owned by this [`Context`].
    pub fn nvvm_mma_shape_attribute<'c>(&'c self, value: MmaShape) -> Result<MmaShapeAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::nvvm()?)?;
        let source = format!("#nvvm.shape<m = {}, n = {}, k = {}>", value.m, value.n, value.k);
        self.parse_attribute(&source)?
            .cast::<MmaShapeAttributeRef>()
            .ok_or_else(|| Error::invalid_argument("invalid nvvm mma shape attribute"))
    }

    /// Creates an NVVM `ldmatrix`/`stmatrix` shape attribute owned by this [`Context`].
    pub fn nvvm_ld_st_matrix_shape_attribute<'c>(
        &'c self,
        value: LdStMatrixShape,
    ) -> Result<LdStMatrixShapeAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::nvvm()?)?;
        let source = format!("#nvvm.ld_st_matrix_shape<m = {}, n = {}>", value.m, value.n);
        self.parse_attribute(&source)?
            .cast::<LdStMatrixShapeAttributeRef>()
            .ok_or_else(|| Error::invalid_argument("invalid nvvm ld/st matrix shape attribute"))
    }

    /// Creates a default NVVM target attribute owned by this [`Context`].
    pub fn nvvm_target_attribute<'c>(&'c self) -> Result<TargetAttributeRef<'c, 't>, Error> {
        self.nvvm_target_attribute_with_options(None, None, None, None, None, None, None)
    }

    /// Creates an NVVM target attribute owned by this [`Context`].
    ///
    /// # Parameters
    ///
    ///   - `optimization_level`: Optional optimization level.
    ///   - `triple`: Optional target triple.
    ///   - `chip`: Optional target chip.
    ///   - `features`: Optional target chip features.
    ///   - `flags`: Optional target-specific flags dictionary.
    ///   - `link`: Optional list of files to link.
    ///   - `verify_target`: Optional target-verification setting.
    pub fn nvvm_target_attribute_with_options<'c>(
        &'c self,
        optimization_level: Option<i32>,
        triple: Option<&str>,
        chip: Option<&str>,
        features: Option<&str>,
        flags: Option<DictionaryAttributeRef<'c, 't>>,
        link: Option<ArrayAttributeRef<'c, 't>>,
        verify_target: Option<bool>,
    ) -> Result<TargetAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::nvvm()?)?;
        let mut fields = Vec::new();
        if let Some(optimization_level) = optimization_level {
            fields.push(format!("O = {optimization_level}"));
        }
        if let Some(triple) = triple {
            fields.push(format!("triple = \"{}\"", escape_string(triple)));
        }
        if let Some(chip) = chip {
            fields.push(format!("chip = \"{}\"", escape_string(chip)));
        }
        if let Some(features) = features {
            fields.push(format!("features = \"{}\"", escape_string(features)));
        }
        if let Some(flags) = flags {
            fields.push(format!("flags = {flags}"));
        }
        if let Some(link) = link {
            fields.push(format!("link = {link}"));
        }
        if let Some(verify_target) = verify_target {
            fields.push(format!("verifyTarget = {verify_target}"));
        }
        let source =
            if fields.is_empty() { "#nvvm.target".to_string() } else { format!("#nvvm.target<{}>", fields.join(", ")) };
        self.parse_attribute(&source)?
            .cast::<TargetAttributeRef>()
            .ok_or_else(|| Error::invalid_argument("invalid nvvm target attribute"))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    macro_rules! nvvm_enum_attribute_tests {
        ($test_name:ident, $constructor:ident, $enum_name:ident, $attribute_name:ident, $first:ident, $second:ident) => {
            paste::paste! {
                #[test]
                fn [<test_ $test_name _attribute>]() {
                    let context = Context::new();
                    let attribute = context.$constructor($enum_name::$first).unwrap();
                    assert_eq!(&context, attribute.context());
                    assert_eq!(attribute.value().unwrap(), $enum_name::$first);
                    assert_eq!($enum_name::from_value(attribute.value().unwrap().value()), Some($enum_name::$first));
                }

                #[test]
                fn [<test_ $test_name _attribute_equality>]() {
                    let context = Context::new();
                    let attribute_1 = context.$constructor($enum_name::$first).unwrap();
                    let attribute_2 = context.$constructor($enum_name::$first).unwrap();
                    assert_eq!(attribute_1, attribute_2);
                    if $enum_name::$first != $enum_name::$second {
                        let attribute_2 = context.$constructor($enum_name::$second).unwrap();
                        assert_ne!(attribute_1, attribute_2);
                    }
                }

                #[test]
                fn [<test_ $test_name _attribute_display_and_debug>]() {
                    let context = Context::new();
                    let attribute = context.$constructor($enum_name::$first).unwrap();
                    let expected = attribute.to_string();
                    test_attribute_display_and_debug(attribute, Box::leak(expected.into_boxed_str()));
                }

                #[test]
                fn [<test_ $test_name _attribute_parsing>]() {
                    let context = Context::new();
                    context.load_dialect(DialectHandle::nvvm().unwrap()).unwrap();
                    let attribute = context.$constructor($enum_name::$first).unwrap();
                    assert_eq!(
                        context.parse_attribute(&attribute.to_string()).unwrap().cast::<$attribute_name>().unwrap(),
                        attribute,
                    );
                }

                #[test]
                fn [<test_ $test_name _attribute_casting>]() {
                    let context = Context::new();
                    let attribute = context.$constructor($enum_name::$first).unwrap();
                    test_attribute_casting(attribute);
                }
            }
        };
    }

    nvvm_enum_attribute_tests!(
        floating_point_rounding_mode,
        nvvm_floating_point_rounding_mode_attribute,
        FloatingPointRoundingMode,
        FloatingPointRoundingModeAttributeRef,
        None,
        NearestEven
    );

    nvvm_enum_attribute_tests!(
        saturation_mode,
        nvvm_saturation_mode_attribute,
        SaturationMode,
        SaturationModeAttributeRef,
        None,
        Satfinite
    );

    nvvm_enum_attribute_tests!(
        cache_eviction_priority,
        nvvm_cache_eviction_priority_attribute,
        CacheEvictionPriority,
        CacheEvictionPriorityAttributeRef,
        EvictNormal,
        EvictFirst
    );

    nvvm_enum_attribute_tests!(
        memory_space,
        nvvm_memory_space_attribute,
        MemorySpace,
        MemorySpaceAttributeRef,
        Generic,
        Global
    );

    nvvm_enum_attribute_tests!(
        mem_scope_kind,
        nvvm_mem_scope_kind_attribute,
        MemScopeKind,
        MemScopeKindAttributeRef,
        Cta,
        Cluster
    );

    nvvm_enum_attribute_tests!(
        shared_space,
        nvvm_shared_space_attribute,
        SharedSpace,
        SharedSpaceAttributeRef,
        Cta,
        Cluster
    );

    nvvm_enum_attribute_tests!(
        mem_order_kind,
        nvvm_mem_order_kind_attribute,
        MemOrderKind,
        MemOrderKindAttributeRef,
        Weak,
        Relaxed
    );

    nvvm_enum_attribute_tests!(
        reduction_kind,
        nvvm_reduction_kind_attribute,
        ReductionKind,
        ReductionKindAttributeRef,
        Add,
        And
    );

    nvvm_enum_attribute_tests!(
        barrier_reduction,
        nvvm_barrier_reduction_attribute,
        BarrierReduction,
        BarrierReductionAttributeRef,
        Popc,
        And
    );

    nvvm_enum_attribute_tests!(proxy_kind, nvvm_proxy_kind_attribute, ProxyKind, ProxyKindAttributeRef, Alias, Async);

    nvvm_enum_attribute_tests!(
        set_max_register_action,
        nvvm_set_max_register_action_attribute,
        SetMaxRegisterAction,
        SetMaxRegisterActionAttributeRef,
        Decrease,
        Increase
    );

    nvvm_enum_attribute_tests!(shfl_kind, nvvm_shfl_kind_attribute, ShflKind, ShflKindAttributeRef, Bfly, Up);

    nvvm_enum_attribute_tests!(
        vote_sync_kind,
        nvvm_vote_sync_kind_attribute,
        VoteSyncKind,
        VoteSyncKindAttributeRef,
        Any,
        All
    );

    nvvm_enum_attribute_tests!(
        permute_mode,
        nvvm_permute_mode_attribute,
        PermuteMode,
        PermuteModeAttributeRef,
        Default,
        F4e
    );

    nvvm_enum_attribute_tests!(
        load_cache_modifier_kind,
        nvvm_load_cache_modifier_kind_attribute,
        LoadCacheModifierKind,
        LoadCacheModifierKindAttributeRef,
        Ca,
        Cg
    );

    nvvm_enum_attribute_tests!(
        mma_b1_operation,
        nvvm_mma_b1_operation_attribute,
        MmaB1Operation,
        MmaB1OperationAttributeRef,
        None,
        XorPopc
    );

    nvvm_enum_attribute_tests!(
        mma_integer_overflow,
        nvvm_mma_integer_overflow_attribute,
        MmaIntegerOverflow,
        MmaIntegerOverflowAttributeRef,
        Satfinite,
        Wrapped
    );

    nvvm_enum_attribute_tests!(mma_kind, nvvm_mma_kind_attribute, MmaKind, MmaKindAttributeRef, F8f6f4, F8f6f4);

    nvvm_enum_attribute_tests!(mma_layout, nvvm_mma_layout_attribute, MmaLayout, MmaLayoutAttributeRef, Row, Col);

    nvvm_enum_attribute_tests!(mma_type, nvvm_mma_type_attribute, MmaType, MmaTypeAttributeRef, F16, F32);

    nvvm_enum_attribute_tests!(mma_fragment, nvvm_mma_fragment_attribute, MmaFragment, MmaFragmentAttributeRef, A, B);

    nvvm_enum_attribute_tests!(
        ld_st_matrix_elt_type,
        nvvm_ld_st_matrix_elt_type_attribute,
        LdStMatrixEltType,
        LdStMatrixEltTypeAttributeRef,
        B16,
        B8
    );

    nvvm_enum_attribute_tests!(
        scale_vec_size,
        nvvm_scale_vec_size_attribute,
        ScaleVecSize,
        ScaleVecSizeAttributeRef,
        X1,
        X2
    );

    nvvm_enum_attribute_tests!(
        block_scale_format,
        nvvm_block_scale_format_attribute,
        BlockScaleFormat,
        BlockScaleFormatAttributeRef,
        Ue8m0,
        Ue4m3
    );

    nvvm_enum_attribute_tests!(
        mma_block_scale_kind,
        nvvm_mma_block_scale_kind_attribute,
        MmaBlockScaleKind,
        MmaBlockScaleKindAttributeRef,
        Mxf8f6f4,
        Mxf4
    );

    nvvm_enum_attribute_tests!(
        tma_load_mode,
        nvvm_tma_load_mode_attribute,
        TmaLoadMode,
        TmaLoadModeAttributeRef,
        Tile,
        Im2col
    );

    nvvm_enum_attribute_tests!(
        tma_store_mode,
        nvvm_tma_store_mode_attribute,
        TmaStoreMode,
        TmaStoreModeAttributeRef,
        Tile,
        Im2col
    );

    nvvm_enum_attribute_tests!(
        cta_group_kind,
        nvvm_cta_group_kind_attribute,
        CtaGroupKind,
        CtaGroupKindAttributeRef,
        Cta1,
        Cta2
    );

    nvvm_enum_attribute_tests!(
        prefetch_cache_level,
        nvvm_prefetch_cache_level_attribute,
        PrefetchCacheLevel,
        PrefetchCacheLevelAttributeRef,
        L1,
        L2
    );

    nvvm_enum_attribute_tests!(
        tma_redux_kind,
        nvvm_tma_redux_kind_attribute,
        TmaReduxKind,
        TmaReduxKindAttributeRef,
        Add,
        Max
    );

    nvvm_enum_attribute_tests!(
        wgmma_scale_in,
        nvvm_wgmma_scale_in_attribute,
        WgmmaScaleIn,
        WgmmaScaleInAttributeRef,
        One,
        One
    );

    nvvm_enum_attribute_tests!(
        wgmma_scale_out,
        nvvm_wgmma_scale_out_attribute,
        WgmmaScaleOut,
        WgmmaScaleOutAttributeRef,
        Zero,
        One
    );

    nvvm_enum_attribute_tests!(wgmma_type, nvvm_wgmma_type_attribute, WgmmaType, WgmmaTypeAttributeRef, F16, Tf32);

    nvvm_enum_attribute_tests!(
        grid_dependency_action,
        nvvm_grid_dependency_action_attribute,
        GridDependencyAction,
        GridDependencyActionAttributeRef,
        Wait,
        LaunchDependents
    );

    nvvm_enum_attribute_tests!(
        match_sync_kind,
        nvvm_match_sync_kind_attribute,
        MatchSyncKind,
        MatchSyncKindAttributeRef,
        Any,
        All
    );

    nvvm_enum_attribute_tests!(
        tcgen05_fence_kind,
        nvvm_tcgen05_fence_kind_attribute,
        Tcgen05FenceKind,
        Tcgen05FenceKindAttributeRef,
        Before,
        After
    );

    nvvm_enum_attribute_tests!(
        tcgen05_wait_kind,
        nvvm_tcgen05_wait_kind_attribute,
        Tcgen05WaitKind,
        Tcgen05WaitKindAttributeRef,
        Load,
        Store
    );

    nvvm_enum_attribute_tests!(
        tcgen05_cp_shape,
        nvvm_tcgen05_cp_shape_attribute,
        Tcgen05CpShape,
        Tcgen05CpShapeAttributeRef,
        Shape128x256b,
        Shape4x256b
    );

    nvvm_enum_attribute_tests!(
        tcgen05_cp_multicast,
        nvvm_tcgen05_cp_multicast_attribute,
        Tcgen05CpMulticast,
        Tcgen05CpMulticastAttributeRef,
        None,
        Warpx20213
    );

    nvvm_enum_attribute_tests!(
        tcgen05_cp_src_format,
        nvvm_tcgen05_cp_src_format_attribute,
        Tcgen05CpSrcFormat,
        Tcgen05CpSrcFormatAttributeRef,
        B6x16P32,
        B4x16P64
    );

    nvvm_enum_attribute_tests!(
        tcgen05_ld_st_shape,
        nvvm_tcgen05_ld_st_shape_attribute,
        Tcgen05LdStShape,
        Tcgen05LdStShapeAttributeRef,
        Shape16x64b,
        Shape16x128b
    );

    nvvm_enum_attribute_tests!(
        dot_accumulate_type,
        nvvm_dot_accumulate_type_attribute,
        DotAccumulateType,
        DotAccumulateTypeAttributeRef,
        Signed,
        Unsigned
    );

    nvvm_enum_attribute_tests!(
        cluster_launch_control_query_type,
        nvvm_cluster_launch_control_query_type_attribute,
        ClusterLaunchControlQueryType,
        ClusterLaunchControlQueryTypeAttributeRef,
        IsCanceled,
        GetFirstCtaIdX
    );

    nvvm_enum_attribute_tests!(
        tcgen05_mma_kind,
        nvvm_tcgen05_mma_kind_attribute,
        Tcgen05MmaKind,
        Tcgen05MmaKindAttributeRef,
        F16,
        Tf32
    );

    nvvm_enum_attribute_tests!(
        tcgen05_mma_collector_operation,
        nvvm_tcgen05_mma_collector_operation_attribute,
        Tcgen05MmaCollectorOperation,
        Tcgen05MmaCollectorOperationAttributeRef,
        Discard,
        Lastuse
    );

    nvvm_enum_attribute_tests!(
        tcgen05_mma_block_scale,
        nvvm_tcgen05_mma_block_scale_attribute,
        Tcgen05MmaBlockScale,
        Tcgen05MmaBlockScaleAttributeRef,
        Default,
        Block16
    );

    nvvm_enum_attribute_tests!(
        tcgen05_mma_collector_b_buffer,
        nvvm_tcgen05_mma_collector_b_buffer_attribute,
        Tcgen05MmaCollectorBBuffer,
        Tcgen05MmaCollectorBBufferAttributeRef,
        B0,
        B1
    );

    nvvm_enum_attribute_tests!(
        tensormap_field,
        nvvm_tensormap_field_attribute,
        TensormapField,
        TensormapFieldAttributeRef,
        GlobalAddress,
        Rank
    );

    nvvm_enum_attribute_tests!(
        tensormap_elemtype,
        nvvm_tensormap_elemtype_attribute,
        TensormapElemtype,
        TensormapElemtypeAttributeRef,
        U8,
        U16
    );

    nvvm_enum_attribute_tests!(
        tensormap_interleave_layout,
        nvvm_tensormap_interleave_layout_attribute,
        TensormapInterleaveLayout,
        TensormapInterleaveLayoutAttributeRef,
        NoInterleave,
        B16
    );

    nvvm_enum_attribute_tests!(
        tensormap_swizzle_mode,
        nvvm_tensormap_swizzle_mode_attribute,
        TensormapSwizzleMode,
        TensormapSwizzleModeAttributeRef,
        NoSwizzling,
        B32
    );

    nvvm_enum_attribute_tests!(
        tensormap_swizzle_atomicity,
        nvvm_tensormap_swizzle_atomicity_attribute,
        TensormapSwizzleAtomicity,
        TensormapSwizzleAtomicityAttributeRef,
        B16,
        B32
    );

    nvvm_enum_attribute_tests!(
        tensormap_fill_mode,
        nvvm_tensormap_fill_mode_attribute,
        TensormapFillMode,
        TensormapFillModeAttributeRef,
        Zero,
        OobNan
    );

    #[test]
    fn test_mma_shape_attribute() {
        let context = Context::new();
        let shape = MmaShape { m: 16, n: 8, k: 32 };
        let attribute = context.nvvm_mma_shape_attribute(shape).unwrap();
        assert_eq!(attribute.value().unwrap(), shape);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_ld_st_matrix_shape_attribute() {
        let context = Context::new();
        let shape = LdStMatrixShape { m: 8, n: 8 };
        let attribute = context.nvvm_ld_st_matrix_shape_attribute(shape).unwrap();
        assert_eq!(attribute.value().unwrap(), shape);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_target_attribute() {
        let context = Context::new();
        let attribute = context.nvvm_target_attribute().unwrap();
        assert_eq!(attribute.to_string(), "#nvvm.target");
        test_attribute_casting(attribute);
        let attribute = context
            .nvvm_target_attribute_with_options(None, None, Some("sm_90"), None, None, None, None)
            .unwrap();
        assert!(attribute.to_string().contains("sm_90"));
    }
}
