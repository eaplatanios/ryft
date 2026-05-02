use ryft_xla_sys::bindings::{
    MlirAttribute, MlirLLVMCConv, MlirLLVMCConv_MlirLLVMCConvAArch64_SVE_VectorCall,
    MlirLLVMCConv_MlirLLVMCConvAArch64_VectorCall, MlirLLVMCConv_MlirLLVMCConvAMDGPU_CS,
    MlirLLVMCConv_MlirLLVMCConvAMDGPU_ES, MlirLLVMCConv_MlirLLVMCConvAMDGPU_GS, MlirLLVMCConv_MlirLLVMCConvAMDGPU_Gfx,
    MlirLLVMCConv_MlirLLVMCConvAMDGPU_HS, MlirLLVMCConv_MlirLLVMCConvAMDGPU_KERNEL,
    MlirLLVMCConv_MlirLLVMCConvAMDGPU_LS, MlirLLVMCConv_MlirLLVMCConvAMDGPU_VS, MlirLLVMCConv_MlirLLVMCConvARM_AAPCS,
    MlirLLVMCConv_MlirLLVMCConvARM_AAPCS_VFP, MlirLLVMCConv_MlirLLVMCConvARM_APCS,
    MlirLLVMCConv_MlirLLVMCConvAVR_BUILTIN, MlirLLVMCConv_MlirLLVMCConvAVR_INTR, MlirLLVMCConv_MlirLLVMCConvAnyReg,
    MlirLLVMCConv_MlirLLVMCConvC, MlirLLVMCConv_MlirLLVMCConvCFGuard_Check, MlirLLVMCConv_MlirLLVMCConvCXX_FAST_TLS,
    MlirLLVMCConv_MlirLLVMCConvCold, MlirLLVMCConv_MlirLLVMCConvDUMMY_HHVM, MlirLLVMCConv_MlirLLVMCConvDUMMY_HHVM_C,
    MlirLLVMCConv_MlirLLVMCConvFast, MlirLLVMCConv_MlirLLVMCConvGHC, MlirLLVMCConv_MlirLLVMCConvHiPE,
    MlirLLVMCConv_MlirLLVMCConvIntel_OCL_BI, MlirLLVMCConv_MlirLLVMCConvM68k_INTR,
    MlirLLVMCConv_MlirLLVMCConvMSP430_BUILTIN, MlirLLVMCConv_MlirLLVMCConvMSP430_INTR,
    MlirLLVMCConv_MlirLLVMCConvPTX_Device, MlirLLVMCConv_MlirLLVMCConvPTX_Kernel,
    MlirLLVMCConv_MlirLLVMCConvPreserveAll, MlirLLVMCConv_MlirLLVMCConvPreserveMost,
    MlirLLVMCConv_MlirLLVMCConvSPIR_FUNC, MlirLLVMCConv_MlirLLVMCConvSPIR_KERNEL, MlirLLVMCConv_MlirLLVMCConvSwift,
    MlirLLVMCConv_MlirLLVMCConvSwiftTail, MlirLLVMCConv_MlirLLVMCConvTail,
    MlirLLVMCConv_MlirLLVMCConvWASM_EmscriptenInvoke, MlirLLVMCConv_MlirLLVMCConvWin64,
    MlirLLVMCConv_MlirLLVMCConvX86_64_SysV, MlirLLVMCConv_MlirLLVMCConvX86_FastCall,
    MlirLLVMCConv_MlirLLVMCConvX86_INTR, MlirLLVMCConv_MlirLLVMCConvX86_RegCall,
    MlirLLVMCConv_MlirLLVMCConvX86_StdCall, MlirLLVMCConv_MlirLLVMCConvX86_ThisCall,
    MlirLLVMCConv_MlirLLVMCConvX86_VectorCall, MlirLLVMComdat, MlirLLVMComdat_MlirLLVMComdatAny,
    MlirLLVMComdat_MlirLLVMComdatExactMatch, MlirLLVMComdat_MlirLLVMComdatLargest,
    MlirLLVMComdat_MlirLLVMComdatNoDeduplicate, MlirLLVMComdat_MlirLLVMComdatSameSize, MlirLLVMLinkage,
    MlirLLVMLinkage_MlirLLVMLinkageAppending, MlirLLVMLinkage_MlirLLVMLinkageAvailableExternally,
    MlirLLVMLinkage_MlirLLVMLinkageCommon, MlirLLVMLinkage_MlirLLVMLinkageExternWeak,
    MlirLLVMLinkage_MlirLLVMLinkageExternal, MlirLLVMLinkage_MlirLLVMLinkageInternal,
    MlirLLVMLinkage_MlirLLVMLinkageLinkonce, MlirLLVMLinkage_MlirLLVMLinkageLinkonceODR,
    MlirLLVMLinkage_MlirLLVMLinkagePrivate, MlirLLVMLinkage_MlirLLVMLinkageWeak,
    MlirLLVMLinkage_MlirLLVMLinkageWeakODR, mlirLLVMCConvAttrGet, mlirLLVMComdatAttrGet, mlirLLVMDINullTypeAttrGet,
    mlirLLVMLinkageAttrGet,
};
use ryft_xla_sys::mlir::dialects::llvm::{
    MlirLlvmFramePointerKind, mlirAttributeIsALlvmAccessGroupAttr, mlirAttributeIsALlvmAddressSpaceAttr,
    mlirAttributeIsALlvmAliasScopeAttr, mlirAttributeIsALlvmAliasScopeDomainAttr, mlirAttributeIsALlvmBlockAddressAttr,
    mlirAttributeIsALlvmBlockTagAttr, mlirAttributeIsALlvmCConvAttr, mlirAttributeIsALlvmComdatAttr,
    mlirAttributeIsALlvmConstantRangeAttr, mlirAttributeIsALlvmDenormalFpEnvAttr,
    mlirAttributeIsALlvmDependentLibrariesAttr, mlirAttributeIsALlvmDereferenceableAttr,
    mlirAttributeIsALlvmDiAnnotationAttr, mlirAttributeIsALlvmDiBasicTypeAttr, mlirAttributeIsALlvmDiCommonBlockAttr,
    mlirAttributeIsALlvmDiCompileUnitAttr, mlirAttributeIsALlvmDiCompositeTypeAttr,
    mlirAttributeIsALlvmDiDerivedTypeAttr, mlirAttributeIsALlvmDiExpressionAttr,
    mlirAttributeIsALlvmDiExpressionElemAttr, mlirAttributeIsALlvmDiFileAttr,
    mlirAttributeIsALlvmDiGenericSubrangeAttr, mlirAttributeIsALlvmDiGlobalVariableAttr,
    mlirAttributeIsALlvmDiGlobalVariableExpressionAttr, mlirAttributeIsALlvmDiImportedEntityAttr,
    mlirAttributeIsALlvmDiLabelAttr, mlirAttributeIsALlvmDiLexicalBlockAttr,
    mlirAttributeIsALlvmDiLexicalBlockFileAttr, mlirAttributeIsALlvmDiLocalVariableAttr,
    mlirAttributeIsALlvmDiModuleAttr, mlirAttributeIsALlvmDiNamespaceAttr, mlirAttributeIsALlvmDiNullTypeAttr,
    mlirAttributeIsALlvmDiStringTypeAttr, mlirAttributeIsALlvmDiSubprogramAttr, mlirAttributeIsALlvmDiSubrangeAttr,
    mlirAttributeIsALlvmDiSubroutineTypeAttr, mlirAttributeIsALlvmDsoLocalEquivalentAttr,
    mlirAttributeIsALlvmFramePointerKindAttr, mlirAttributeIsALlvmLinkageAttr, mlirAttributeIsALlvmLoopAnnotationAttr,
    mlirAttributeIsALlvmLoopDistributeAttr, mlirAttributeIsALlvmLoopInterleaveAttr, mlirAttributeIsALlvmLoopLicmAttr,
    mlirAttributeIsALlvmLoopPeeledAttr, mlirAttributeIsALlvmLoopPipelineAttr, mlirAttributeIsALlvmLoopUnrollAndJamAttr,
    mlirAttributeIsALlvmLoopUnrollAttr, mlirAttributeIsALlvmLoopUnswitchAttr, mlirAttributeIsALlvmLoopVectorizeAttr,
    mlirAttributeIsALlvmMdConstantAttr, mlirAttributeIsALlvmMdFuncAttr, mlirAttributeIsALlvmMdNodeAttr,
    mlirAttributeIsALlvmMdStringAttr, mlirAttributeIsALlvmMemoryEffectsAttr, mlirAttributeIsALlvmMmraTagAttr,
    mlirAttributeIsALlvmModuleFlagAttr, mlirAttributeIsALlvmModuleFlagCgProfileEntryAttr,
    mlirAttributeIsALlvmModuleFlagProfileSummaryAttr, mlirAttributeIsALlvmModuleFlagProfileSummaryDetailedAttr,
    mlirAttributeIsALlvmPoisonAttr, mlirAttributeIsALlvmTailCallKindAttr, mlirAttributeIsALlvmTargetAttr,
    mlirAttributeIsALlvmTargetFeaturesAttr, mlirAttributeIsALlvmTbaaMemberAttr, mlirAttributeIsALlvmTbaaRootAttr,
    mlirAttributeIsALlvmTbaaTagAttr, mlirAttributeIsALlvmTbaaTypeDescriptorAttr, mlirAttributeIsALlvmUndefAttr,
    mlirAttributeIsALlvmUwTableKindAttr, mlirAttributeIsALlvmVScaleRangeAttr, mlirAttributeIsALlvmVecTypeHintAttr,
    mlirAttributeIsALlvmWorkgroupAttributionAttr, mlirAttributeIsALlvmZeroAttr, mlirLLVMMDConstantAttrGet,
    mlirLLVMMDConstantAttrGetValue, mlirLLVMMDFuncAttrGet, mlirLLVMMDFuncAttrGetName, mlirLLVMMDNodeAttrGet,
    mlirLLVMMDNodeAttrGetNumOperands, mlirLLVMMDNodeAttrGetOperand, mlirLLVMMDStringAttrGet,
    mlirLLVMMDStringAttrGetValue, mlirLlvmAddressSpaceAttrGet, mlirLlvmAddressSpaceAttrGetAddressSpace,
    mlirLlvmCConvAttrGetValue, mlirLlvmComdatAttrGetValue, mlirLlvmFramePointerKindAttrGet,
    mlirLlvmFramePointerKindAttrGetValue, mlirLlvmLinkageAttrGetValue, mlirLlvmPoisonAttrGet, mlirLlvmUndefAttrGet,
    mlirLlvmZeroAttrGet,
};

use crate::{
    Attribute, AttributeRef, Context, DialectHandle, FlatSymbolRefAttributeRef, FromWithContext, StringRef,
    mlir_subtype_trait_impls,
};

macro_rules! llvm_enum_attribute {
    (
        $attribute_name:ident,
        $enum_name:ident,
        $context_method:ident,
        $is_a:path,
        $get:path,
        $get_value:path,
        $description:literal $(,)*
    ) => {
        #[doc = "LLVM "]
        #[doc = $description]
        #[doc = " [`Attribute`]."]
        #[derive(Copy, Clone)]
        pub struct $attribute_name<'c, 't> {
            /// Handle that represents this [`Attribute`] in the MLIR C API.
            handle: MlirAttribute,

            /// [`Context`] that owns this [`Attribute`].
            context: &'c Context<'t>,
        }

        impl $attribute_name<'_, '_> {
            #[doc = "Returns the LLVM "]
            #[doc = $description]
            #[doc = " stored in this attribute."]
            pub fn value(&self) -> $enum_name {
                $enum_name::from_c_api(unsafe { $get_value(self.handle) })
                    .expect(concat!("invalid LLVM ", $description, " attribute value"))
            }
        }

        impl<'c, 't> Attribute<'c, 't> for $attribute_name<'c, 't> {
            unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
                if handle.ptr.is_null() {
                    return None;
                }
                if unsafe { $is_a(handle) } {
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
            #[doc = "Creates a new LLVM "]
            #[doc = $description]
            #[doc = " attribute owned by this [`Context`]."]
            pub fn $context_method<'c>(&'c self, value: $enum_name) -> $attribute_name<'c, 't> {
                self.load_dialect(DialectHandle::llvm());
                unsafe {
                    $attribute_name::from_c_api($get(*self.handle.borrow(), value.to_c_api()), self)
                        .expect(concat!("invalid arguments to `Context::", stringify!($context_method), "`"))
                }
            }
        }
    };
}

macro_rules! llvm_attribute {
    ($attribute_name:ident, $is_a:path, $description:literal $(,)*) => {
        #[doc = "LLVM "]
        #[doc = $description]
        #[doc = " [`Attribute`]."]
        #[derive(Copy, Clone)]
        pub struct $attribute_name<'c, 't> {
            /// Handle that represents this [`Attribute`] in the MLIR C API.
            handle: MlirAttribute,

            /// [`Context`] that owns this [`Attribute`].
            context: &'c Context<'t>,
        }

        impl<'c, 't> Attribute<'c, 't> for $attribute_name<'c, 't> {
            unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
                if handle.ptr.is_null() {
                    return None;
                }
                if unsafe { $is_a(handle) } {
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
    };
}

llvm_attribute!(AddressSpaceAttributeRef, mlirAttributeIsALlvmAddressSpaceAttr, "address space");

/// LLVM calling convention.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CallingConvention {
    /// C calling convention.
    C,

    /// Fast calling convention.
    Fast,

    /// Cold calling convention.
    Cold,

    /// Glasgow Haskell Compiler calling convention.
    Ghc,

    /// High Performance Erlang calling convention.
    HiPe,

    /// Any-register calling convention.
    AnyReg,

    /// Preserve-most calling convention.
    PreserveMost,

    /// Preserve-all calling convention.
    PreserveAll,

    /// Swift calling convention.
    Swift,

    /// C++ fast thread-local storage calling convention.
    CxxFastTls,

    /// Tail calling convention.
    Tail,

    /// Control Flow Guard check calling convention.
    CfGuardCheck,

    /// Swift tail calling convention.
    SwiftTail,

    /// x86 stdcall calling convention.
    X86StdCall,

    /// x86 fastcall calling convention.
    X86FastCall,

    /// ARM APCS calling convention.
    ArmApcs,

    /// ARM AAPCS calling convention.
    ArmAapcs,

    /// ARM AAPCS VFP calling convention.
    ArmAapcsVfp,

    /// MSP430 interrupt calling convention.
    Msp430Intr,

    /// x86 thiscall calling convention.
    X86ThisCall,

    /// PTX kernel calling convention.
    PtxKernel,

    /// PTX device calling convention.
    PtxDevice,

    /// SPIR function calling convention.
    SpirFunc,

    /// SPIR kernel calling convention.
    SpirKernel,

    /// Intel OpenCL built-in calling convention.
    IntelOclBi,

    /// x86-64 System V calling convention.
    X86_64SysV,

    /// Win64 calling convention.
    Win64,

    /// x86 vectorcall calling convention.
    X86VectorCall,

    /// HHVM placeholder calling convention.
    DummyHhvm,

    /// HHVM C placeholder calling convention.
    DummyHhvmC,

    /// x86 interrupt calling convention.
    X86Intr,

    /// AVR interrupt calling convention.
    AvrIntr,

    /// AVR built-in calling convention.
    AvrBuiltin,

    /// AMDGPU vertex shader calling convention.
    AmdGpuVs,

    /// AMDGPU geometry shader calling convention.
    AmdGpuGs,

    /// AMDGPU compute shader calling convention.
    AmdGpuCs,

    /// AMDGPU kernel calling convention.
    AmdGpuKernel,

    /// x86 regcall calling convention.
    X86RegCall,

    /// AMDGPU hull shader calling convention.
    AmdGpuHs,

    /// MSP430 built-in calling convention.
    Msp430Builtin,

    /// AMDGPU local shader calling convention.
    AmdGpuLs,

    /// AMDGPU export shader calling convention.
    AmdGpuEs,

    /// AArch64 vectorcall calling convention.
    AArch64VectorCall,

    /// AArch64 SVE vectorcall calling convention.
    AArch64SveVectorCall,

    /// WebAssembly Emscripten invoke calling convention.
    WasmEmscriptenInvoke,

    /// AMDGPU graphics calling convention.
    AmdGpuGfx,

    /// M68k interrupt calling convention.
    M68kIntr,
}

impl CallingConvention {
    /// All LLVM calling convention variants exposed by the MLIR C API.
    pub const ALL: &'static [Self] = &[
        Self::C,
        Self::Fast,
        Self::Cold,
        Self::Ghc,
        Self::HiPe,
        Self::AnyReg,
        Self::PreserveMost,
        Self::PreserveAll,
        Self::Swift,
        Self::CxxFastTls,
        Self::Tail,
        Self::CfGuardCheck,
        Self::SwiftTail,
        Self::X86StdCall,
        Self::X86FastCall,
        Self::ArmApcs,
        Self::ArmAapcs,
        Self::ArmAapcsVfp,
        Self::Msp430Intr,
        Self::X86ThisCall,
        Self::PtxKernel,
        Self::PtxDevice,
        Self::SpirFunc,
        Self::SpirKernel,
        Self::IntelOclBi,
        Self::X86_64SysV,
        Self::Win64,
        Self::X86VectorCall,
        Self::DummyHhvm,
        Self::DummyHhvmC,
        Self::X86Intr,
        Self::AvrIntr,
        Self::AvrBuiltin,
        Self::AmdGpuVs,
        Self::AmdGpuGs,
        Self::AmdGpuCs,
        Self::AmdGpuKernel,
        Self::X86RegCall,
        Self::AmdGpuHs,
        Self::Msp430Builtin,
        Self::AmdGpuLs,
        Self::AmdGpuEs,
        Self::AArch64VectorCall,
        Self::AArch64SveVectorCall,
        Self::WasmEmscriptenInvoke,
        Self::AmdGpuGfx,
        Self::M68kIntr,
    ];

    /// Returns the MLIR C API representation of this calling convention.
    pub fn to_c_api(&self) -> MlirLLVMCConv {
        match self {
            Self::C => MlirLLVMCConv_MlirLLVMCConvC,
            Self::Fast => MlirLLVMCConv_MlirLLVMCConvFast,
            Self::Cold => MlirLLVMCConv_MlirLLVMCConvCold,
            Self::Ghc => MlirLLVMCConv_MlirLLVMCConvGHC,
            Self::HiPe => MlirLLVMCConv_MlirLLVMCConvHiPE,
            Self::AnyReg => MlirLLVMCConv_MlirLLVMCConvAnyReg,
            Self::PreserveMost => MlirLLVMCConv_MlirLLVMCConvPreserveMost,
            Self::PreserveAll => MlirLLVMCConv_MlirLLVMCConvPreserveAll,
            Self::Swift => MlirLLVMCConv_MlirLLVMCConvSwift,
            Self::CxxFastTls => MlirLLVMCConv_MlirLLVMCConvCXX_FAST_TLS,
            Self::Tail => MlirLLVMCConv_MlirLLVMCConvTail,
            Self::CfGuardCheck => MlirLLVMCConv_MlirLLVMCConvCFGuard_Check,
            Self::SwiftTail => MlirLLVMCConv_MlirLLVMCConvSwiftTail,
            Self::X86StdCall => MlirLLVMCConv_MlirLLVMCConvX86_StdCall,
            Self::X86FastCall => MlirLLVMCConv_MlirLLVMCConvX86_FastCall,
            Self::ArmApcs => MlirLLVMCConv_MlirLLVMCConvARM_APCS,
            Self::ArmAapcs => MlirLLVMCConv_MlirLLVMCConvARM_AAPCS,
            Self::ArmAapcsVfp => MlirLLVMCConv_MlirLLVMCConvARM_AAPCS_VFP,
            Self::Msp430Intr => MlirLLVMCConv_MlirLLVMCConvMSP430_INTR,
            Self::X86ThisCall => MlirLLVMCConv_MlirLLVMCConvX86_ThisCall,
            Self::PtxKernel => MlirLLVMCConv_MlirLLVMCConvPTX_Kernel,
            Self::PtxDevice => MlirLLVMCConv_MlirLLVMCConvPTX_Device,
            Self::SpirFunc => MlirLLVMCConv_MlirLLVMCConvSPIR_FUNC,
            Self::SpirKernel => MlirLLVMCConv_MlirLLVMCConvSPIR_KERNEL,
            Self::IntelOclBi => MlirLLVMCConv_MlirLLVMCConvIntel_OCL_BI,
            Self::X86_64SysV => MlirLLVMCConv_MlirLLVMCConvX86_64_SysV,
            Self::Win64 => MlirLLVMCConv_MlirLLVMCConvWin64,
            Self::X86VectorCall => MlirLLVMCConv_MlirLLVMCConvX86_VectorCall,
            Self::DummyHhvm => MlirLLVMCConv_MlirLLVMCConvDUMMY_HHVM,
            Self::DummyHhvmC => MlirLLVMCConv_MlirLLVMCConvDUMMY_HHVM_C,
            Self::X86Intr => MlirLLVMCConv_MlirLLVMCConvX86_INTR,
            Self::AvrIntr => MlirLLVMCConv_MlirLLVMCConvAVR_INTR,
            Self::AvrBuiltin => MlirLLVMCConv_MlirLLVMCConvAVR_BUILTIN,
            Self::AmdGpuVs => MlirLLVMCConv_MlirLLVMCConvAMDGPU_VS,
            Self::AmdGpuGs => MlirLLVMCConv_MlirLLVMCConvAMDGPU_GS,
            Self::AmdGpuCs => MlirLLVMCConv_MlirLLVMCConvAMDGPU_CS,
            Self::AmdGpuKernel => MlirLLVMCConv_MlirLLVMCConvAMDGPU_KERNEL,
            Self::X86RegCall => MlirLLVMCConv_MlirLLVMCConvX86_RegCall,
            Self::AmdGpuHs => MlirLLVMCConv_MlirLLVMCConvAMDGPU_HS,
            Self::Msp430Builtin => MlirLLVMCConv_MlirLLVMCConvMSP430_BUILTIN,
            Self::AmdGpuLs => MlirLLVMCConv_MlirLLVMCConvAMDGPU_LS,
            Self::AmdGpuEs => MlirLLVMCConv_MlirLLVMCConvAMDGPU_ES,
            Self::AArch64VectorCall => MlirLLVMCConv_MlirLLVMCConvAArch64_VectorCall,
            Self::AArch64SveVectorCall => MlirLLVMCConv_MlirLLVMCConvAArch64_SVE_VectorCall,
            Self::WasmEmscriptenInvoke => MlirLLVMCConv_MlirLLVMCConvWASM_EmscriptenInvoke,
            Self::AmdGpuGfx => MlirLLVMCConv_MlirLLVMCConvAMDGPU_Gfx,
            Self::M68kIntr => MlirLLVMCConv_MlirLLVMCConvM68k_INTR,
        }
    }

    /// Returns the [`CallingConvention`] represented by `value`.
    pub fn from_c_api(value: MlirLLVMCConv) -> Option<Self> {
        Self::ALL.iter().copied().find(|candidate| candidate.to_c_api() == value)
    }
}

llvm_enum_attribute!(
    CallingConventionAttributeRef,
    CallingConvention,
    llvm_calling_convention_attribute,
    mlirAttributeIsALlvmCConvAttr,
    mlirLLVMCConvAttrGet,
    mlirLlvmCConvAttrGetValue,
    "calling convention",
);

/// LLVM comdat selector kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Comdat {
    /// Allows the linker to choose any matching comdat.
    Any,

    /// Requires exact content match.
    ExactMatch,

    /// Keeps the largest matching comdat.
    Largest,

    /// Disables deduplication.
    NoDeduplicate,

    /// Keeps same-size matching comdats.
    SameSize,
}

impl Comdat {
    /// All LLVM comdat selector variants exposed by the MLIR C API.
    pub const ALL: &'static [Self] = &[Self::Any, Self::ExactMatch, Self::Largest, Self::NoDeduplicate, Self::SameSize];

    /// Returns the MLIR C API representation of this comdat selector.
    pub fn to_c_api(&self) -> MlirLLVMComdat {
        match self {
            Self::Any => MlirLLVMComdat_MlirLLVMComdatAny,
            Self::ExactMatch => MlirLLVMComdat_MlirLLVMComdatExactMatch,
            Self::Largest => MlirLLVMComdat_MlirLLVMComdatLargest,
            Self::NoDeduplicate => MlirLLVMComdat_MlirLLVMComdatNoDeduplicate,
            Self::SameSize => MlirLLVMComdat_MlirLLVMComdatSameSize,
        }
    }

    /// Returns the [`Comdat`] represented by `value`.
    pub fn from_c_api(value: MlirLLVMComdat) -> Option<Self> {
        Self::ALL.iter().copied().find(|candidate| candidate.to_c_api() == value)
    }
}

llvm_enum_attribute!(
    ComdatAttributeRef,
    Comdat,
    llvm_comdat_attribute,
    mlirAttributeIsALlvmComdatAttr,
    mlirLLVMComdatAttrGet,
    mlirLlvmComdatAttrGetValue,
    "comdat selector",
);

/// LLVM symbol linkage.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Linkage {
    /// Externally visible definition or declaration.
    External,

    /// Definition available only for optimization and not emitted.
    AvailableExternally,

    /// Link-once definition.
    Linkonce,

    /// Link-once ODR definition.
    LinkonceOdr,

    /// Weak definition.
    Weak,

    /// Weak ODR definition.
    WeakOdr,

    /// Appending linkage for special globals.
    Appending,

    /// Internal linkage.
    Internal,

    /// Private linkage.
    Private,

    /// External weak declaration.
    ExternWeak,

    /// Common tentative definition.
    Common,
}

impl Linkage {
    /// All LLVM linkage variants exposed by the MLIR C API.
    pub const ALL: &'static [Self] = &[
        Self::External,
        Self::AvailableExternally,
        Self::Linkonce,
        Self::LinkonceOdr,
        Self::Weak,
        Self::WeakOdr,
        Self::Appending,
        Self::Internal,
        Self::Private,
        Self::ExternWeak,
        Self::Common,
    ];

    /// Returns the MLIR C API representation of this linkage.
    pub fn to_c_api(&self) -> MlirLLVMLinkage {
        match self {
            Self::External => MlirLLVMLinkage_MlirLLVMLinkageExternal,
            Self::AvailableExternally => MlirLLVMLinkage_MlirLLVMLinkageAvailableExternally,
            Self::Linkonce => MlirLLVMLinkage_MlirLLVMLinkageLinkonce,
            Self::LinkonceOdr => MlirLLVMLinkage_MlirLLVMLinkageLinkonceODR,
            Self::Weak => MlirLLVMLinkage_MlirLLVMLinkageWeak,
            Self::WeakOdr => MlirLLVMLinkage_MlirLLVMLinkageWeakODR,
            Self::Appending => MlirLLVMLinkage_MlirLLVMLinkageAppending,
            Self::Internal => MlirLLVMLinkage_MlirLLVMLinkageInternal,
            Self::Private => MlirLLVMLinkage_MlirLLVMLinkagePrivate,
            Self::ExternWeak => MlirLLVMLinkage_MlirLLVMLinkageExternWeak,
            Self::Common => MlirLLVMLinkage_MlirLLVMLinkageCommon,
        }
    }

    /// Returns the [`Linkage`] represented by `value`.
    pub fn from_c_api(value: MlirLLVMLinkage) -> Option<Self> {
        Self::ALL.iter().copied().find(|candidate| candidate.to_c_api() == value)
    }
}

llvm_enum_attribute!(
    LinkageAttributeRef,
    Linkage,
    llvm_linkage_attribute,
    mlirAttributeIsALlvmLinkageAttr,
    mlirLLVMLinkageAttrGet,
    mlirLlvmLinkageAttrGetValue,
    "linkage",
);

llvm_attribute!(FramePointerKindAttributeRef, mlirAttributeIsALlvmFramePointerKindAttr, "frame pointer kind");
llvm_attribute!(LoopVectorizeAttributeRef, mlirAttributeIsALlvmLoopVectorizeAttr, "loop vectorize metadata");
llvm_attribute!(LoopInterleaveAttributeRef, mlirAttributeIsALlvmLoopInterleaveAttr, "loop interleave metadata");
llvm_attribute!(LoopUnrollAttributeRef, mlirAttributeIsALlvmLoopUnrollAttr, "loop unroll metadata");
llvm_attribute!(LoopUnrollAndJamAttributeRef, mlirAttributeIsALlvmLoopUnrollAndJamAttr, "loop unroll-and-jam metadata");
llvm_attribute!(LoopLicmAttributeRef, mlirAttributeIsALlvmLoopLicmAttr, "loop LICM metadata");
llvm_attribute!(LoopDistributeAttributeRef, mlirAttributeIsALlvmLoopDistributeAttr, "loop distribute metadata");
llvm_attribute!(LoopPipelineAttributeRef, mlirAttributeIsALlvmLoopPipelineAttr, "loop pipeline metadata");
llvm_attribute!(LoopPeeledAttributeRef, mlirAttributeIsALlvmLoopPeeledAttr, "loop peeled metadata");
llvm_attribute!(LoopUnswitchAttributeRef, mlirAttributeIsALlvmLoopUnswitchAttr, "loop unswitch metadata");
llvm_attribute!(LoopAnnotationAttributeRef, mlirAttributeIsALlvmLoopAnnotationAttr, "loop annotation metadata");
llvm_attribute!(
    DiExpressionElemAttributeRef,
    mlirAttributeIsALlvmDiExpressionElemAttr,
    "debug info expression element",
);
llvm_attribute!(DiExpressionAttributeRef, mlirAttributeIsALlvmDiExpressionAttr, "debug info expression");
llvm_attribute!(DiNullTypeAttributeRef, mlirAttributeIsALlvmDiNullTypeAttr, "debug info null type");
llvm_attribute!(DiBasicTypeAttributeRef, mlirAttributeIsALlvmDiBasicTypeAttr, "debug info basic type");
llvm_attribute!(DiCompileUnitAttributeRef, mlirAttributeIsALlvmDiCompileUnitAttr, "debug info compile unit");
llvm_attribute!(DiCompositeTypeAttributeRef, mlirAttributeIsALlvmDiCompositeTypeAttr, "debug info composite type");
llvm_attribute!(DiDerivedTypeAttributeRef, mlirAttributeIsALlvmDiDerivedTypeAttr, "debug info derived type");
llvm_attribute!(DiFileAttributeRef, mlirAttributeIsALlvmDiFileAttr, "debug info file");
llvm_attribute!(
    DiGlobalVariableExpressionAttributeRef,
    mlirAttributeIsALlvmDiGlobalVariableExpressionAttr,
    "debug info global variable expression",
);
llvm_attribute!(DiGlobalVariableAttributeRef, mlirAttributeIsALlvmDiGlobalVariableAttr, "debug info global variable");
llvm_attribute!(DiLexicalBlockAttributeRef, mlirAttributeIsALlvmDiLexicalBlockAttr, "debug info lexical block");
llvm_attribute!(
    DiLexicalBlockFileAttributeRef,
    mlirAttributeIsALlvmDiLexicalBlockFileAttr,
    "debug info lexical block file",
);
llvm_attribute!(DiLocalVariableAttributeRef, mlirAttributeIsALlvmDiLocalVariableAttr, "debug info local variable");
llvm_attribute!(DiSubprogramAttributeRef, mlirAttributeIsALlvmDiSubprogramAttr, "debug info subprogram");
llvm_attribute!(DiModuleAttributeRef, mlirAttributeIsALlvmDiModuleAttr, "debug info module");
llvm_attribute!(DiNamespaceAttributeRef, mlirAttributeIsALlvmDiNamespaceAttr, "debug info namespace");
llvm_attribute!(DiImportedEntityAttributeRef, mlirAttributeIsALlvmDiImportedEntityAttr, "debug info imported entity");
llvm_attribute!(DiAnnotationAttributeRef, mlirAttributeIsALlvmDiAnnotationAttr, "debug info annotation");
llvm_attribute!(DiSubrangeAttributeRef, mlirAttributeIsALlvmDiSubrangeAttr, "debug info subrange");
llvm_attribute!(DiCommonBlockAttributeRef, mlirAttributeIsALlvmDiCommonBlockAttr, "debug info common block");
llvm_attribute!(
    DiGenericSubrangeAttributeRef,
    mlirAttributeIsALlvmDiGenericSubrangeAttr,
    "debug info generic subrange",
);
llvm_attribute!(DiSubroutineTypeAttributeRef, mlirAttributeIsALlvmDiSubroutineTypeAttr, "debug info subroutine type");
llvm_attribute!(DiLabelAttributeRef, mlirAttributeIsALlvmDiLabelAttr, "debug info label");
llvm_attribute!(DiStringTypeAttributeRef, mlirAttributeIsALlvmDiStringTypeAttr, "debug info string type");
llvm_attribute!(MemoryEffectsAttributeRef, mlirAttributeIsALlvmMemoryEffectsAttr, "memory effects");
llvm_attribute!(
    DenormalFpEnvAttributeRef,
    mlirAttributeIsALlvmDenormalFpEnvAttr,
    "denormal floating-point environment",
);
llvm_attribute!(AliasScopeDomainAttributeRef, mlirAttributeIsALlvmAliasScopeDomainAttr, "alias scope domain");
llvm_attribute!(AliasScopeAttributeRef, mlirAttributeIsALlvmAliasScopeAttr, "alias scope");
llvm_attribute!(AccessGroupAttributeRef, mlirAttributeIsALlvmAccessGroupAttr, "access group");
llvm_attribute!(TbaaRootAttributeRef, mlirAttributeIsALlvmTbaaRootAttr, "TBAA root");
llvm_attribute!(TbaaMemberAttributeRef, mlirAttributeIsALlvmTbaaMemberAttr, "TBAA member");
llvm_attribute!(TbaaTypeDescriptorAttributeRef, mlirAttributeIsALlvmTbaaTypeDescriptorAttr, "TBAA type descriptor");
llvm_attribute!(TbaaTagAttributeRef, mlirAttributeIsALlvmTbaaTagAttr, "TBAA tag");
llvm_attribute!(MmraTagAttributeRef, mlirAttributeIsALlvmMmraTagAttr, "MMRA tag");
llvm_attribute!(ConstantRangeAttributeRef, mlirAttributeIsALlvmConstantRangeAttr, "constant range");
llvm_attribute!(VScaleRangeAttributeRef, mlirAttributeIsALlvmVScaleRangeAttr, "vscale range");
llvm_attribute!(TargetFeaturesAttributeRef, mlirAttributeIsALlvmTargetFeaturesAttr, "target features");
llvm_attribute!(TargetAttributeRef, mlirAttributeIsALlvmTargetAttr, "target");
llvm_attribute!(UndefAttributeRef, mlirAttributeIsALlvmUndefAttr, "undef");
llvm_attribute!(PoisonAttributeRef, mlirAttributeIsALlvmPoisonAttr, "poison");
llvm_attribute!(DsoLocalEquivalentAttributeRef, mlirAttributeIsALlvmDsoLocalEquivalentAttr, "DSO local equivalent");
llvm_attribute!(BlockTagAttributeRef, mlirAttributeIsALlvmBlockTagAttr, "block tag");
llvm_attribute!(BlockAddressAttributeRef, mlirAttributeIsALlvmBlockAddressAttr, "block address");
llvm_attribute!(VecTypeHintAttributeRef, mlirAttributeIsALlvmVecTypeHintAttr, "vector type hint");
llvm_attribute!(ZeroAttributeRef, mlirAttributeIsALlvmZeroAttr, "zero");
llvm_attribute!(TailCallKindAttributeRef, mlirAttributeIsALlvmTailCallKindAttr, "tail-call kind");
llvm_attribute!(
    WorkgroupAttributionAttributeRef,
    mlirAttributeIsALlvmWorkgroupAttributionAttr,
    "workgroup attribution",
);
llvm_attribute!(DereferenceableAttributeRef, mlirAttributeIsALlvmDereferenceableAttr, "dereferenceable metadata");
llvm_attribute!(ModuleFlagAttributeRef, mlirAttributeIsALlvmModuleFlagAttr, "module flag");
llvm_attribute!(
    ModuleFlagCgProfileEntryAttributeRef,
    mlirAttributeIsALlvmModuleFlagCgProfileEntryAttr,
    "module flag CG profile entry",
);
llvm_attribute!(
    ModuleFlagProfileSummaryDetailedAttributeRef,
    mlirAttributeIsALlvmModuleFlagProfileSummaryDetailedAttr,
    "module flag profile summary detail",
);
llvm_attribute!(
    ModuleFlagProfileSummaryAttributeRef,
    mlirAttributeIsALlvmModuleFlagProfileSummaryAttr,
    "module flag profile summary",
);
llvm_attribute!(DependentLibrariesAttributeRef, mlirAttributeIsALlvmDependentLibrariesAttr, "dependent libraries");
llvm_attribute!(UwTableKindAttributeRef, mlirAttributeIsALlvmUwTableKindAttr, "unwind table kind");
llvm_attribute!(MdStringAttributeRef, mlirAttributeIsALlvmMdStringAttr, "metadata string");
llvm_attribute!(MdConstantAttributeRef, mlirAttributeIsALlvmMdConstantAttr, "metadata constant");
llvm_attribute!(MdFuncAttributeRef, mlirAttributeIsALlvmMdFuncAttr, "metadata function");
llvm_attribute!(MdNodeAttributeRef, mlirAttributeIsALlvmMdNodeAttr, "metadata node");

impl AddressSpaceAttributeRef<'_, '_> {
    /// Returns the LLVM address space number.
    pub fn address_space(&self) -> u32 {
        unsafe { mlirLlvmAddressSpaceAttrGetAddressSpace(self.handle) }
    }
}

/// LLVM frame pointer kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FramePointerKind {
    /// No frame pointer is requested.
    None,

    /// A frame pointer is requested for non-leaf functions.
    NonLeaf,

    /// A frame pointer is requested for all functions.
    All,

    /// The frame pointer register is reserved.
    Reserved,

    /// A frame pointer is requested for non-leaf functions without reserving it.
    NonLeafNoReserve,
}

impl FramePointerKind {
    /// Returns the MLIR C API representation of this frame pointer kind.
    pub fn to_c_api(&self) -> MlirLlvmFramePointerKind {
        match self {
            Self::None => MlirLlvmFramePointerKind::MLIR_LLVM_FRAME_POINTER_KIND_NONE,
            Self::NonLeaf => MlirLlvmFramePointerKind::MLIR_LLVM_FRAME_POINTER_KIND_NON_LEAF,
            Self::All => MlirLlvmFramePointerKind::MLIR_LLVM_FRAME_POINTER_KIND_ALL,
            Self::Reserved => MlirLlvmFramePointerKind::MLIR_LLVM_FRAME_POINTER_KIND_RESERVED,
            Self::NonLeafNoReserve => MlirLlvmFramePointerKind::MLIR_LLVM_FRAME_POINTER_KIND_NON_LEAF_NO_RESERVE,
        }
    }

    /// Returns the [`FramePointerKind`] represented by `value`.
    pub fn from_c_api(value: MlirLlvmFramePointerKind) -> Self {
        match value {
            MlirLlvmFramePointerKind::MLIR_LLVM_FRAME_POINTER_KIND_NONE => Self::None,
            MlirLlvmFramePointerKind::MLIR_LLVM_FRAME_POINTER_KIND_NON_LEAF => Self::NonLeaf,
            MlirLlvmFramePointerKind::MLIR_LLVM_FRAME_POINTER_KIND_ALL => Self::All,
            MlirLlvmFramePointerKind::MLIR_LLVM_FRAME_POINTER_KIND_RESERVED => Self::Reserved,
            MlirLlvmFramePointerKind::MLIR_LLVM_FRAME_POINTER_KIND_NON_LEAF_NO_RESERVE => Self::NonLeafNoReserve,
        }
    }
}

impl FramePointerKindAttributeRef<'_, '_> {
    /// Returns the LLVM frame pointer kind stored in this attribute.
    pub fn value(&self) -> FramePointerKind {
        FramePointerKind::from_c_api(unsafe { mlirLlvmFramePointerKindAttrGetValue(self.handle) })
    }
}

impl<'c> MdStringAttributeRef<'c, '_> {
    /// Returns the metadata string value.
    pub fn value(&self) -> StringRef<'c> {
        unsafe { StringRef::from_c_api(mlirLLVMMDStringAttrGetValue(self.handle)) }
    }
}

impl<'c, 't> MdConstantAttributeRef<'c, 't> {
    /// Returns the metadata constant value.
    pub fn value(&self) -> AttributeRef<'c, 't> {
        unsafe { AttributeRef::from_c_api(mlirLLVMMDConstantAttrGetValue(self.handle), self.context).unwrap() }
    }
}

impl<'c, 't> MdFuncAttributeRef<'c, 't> {
    /// Returns the referenced metadata function symbol.
    pub fn name(&self) -> FlatSymbolRefAttributeRef<'c, 't> {
        unsafe { FlatSymbolRefAttributeRef::from_c_api(mlirLLVMMDFuncAttrGetName(self.handle), self.context).unwrap() }
    }
}

impl<'c, 't> MdNodeAttributeRef<'c, 't> {
    /// Returns the number of metadata node operands.
    pub fn operand_count(&self) -> usize {
        usize::try_from(unsafe { mlirLLVMMDNodeAttrGetNumOperands(self.handle) })
            .expect("invalid `#llvm.md_node` operand count")
    }

    /// Returns the metadata node operands.
    pub fn operands(&self) -> impl Iterator<Item = AttributeRef<'c, 't>> {
        (0..self.operand_count()).map(|index| self.operand(index))
    }

    /// Returns the `index`-th metadata node operand.
    pub fn operand(&self, index: usize) -> AttributeRef<'c, 't> {
        if index >= self.operand_count() {
            panic!("LLVM metadata node operand index is out of bounds");
        }
        unsafe {
            AttributeRef::from_c_api(mlirLLVMMDNodeAttrGetOperand(self.handle, index.cast_signed()), self.context)
                .expect("invalid `#llvm.md_node` operand")
        }
    }
}

impl<'t> Context<'t> {
    /// Creates a new LLVM [`AddressSpaceAttributeRef`] owned by this [`Context`].
    pub fn llvm_address_space_attribute<'c>(&'c self, address_space: u32) -> AddressSpaceAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        unsafe {
            AddressSpaceAttributeRef::from_c_api(
                mlirLlvmAddressSpaceAttrGet(*self.handle.borrow(), address_space),
                self,
            )
            .expect("invalid LLVM address space attribute")
        }
    }

    /// Creates a new LLVM [`FramePointerKindAttributeRef`] owned by this [`Context`].
    pub fn llvm_frame_pointer_kind_attribute<'c>(
        &'c self,
        value: FramePointerKind,
    ) -> FramePointerKindAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        unsafe {
            FramePointerKindAttributeRef::from_c_api(
                mlirLlvmFramePointerKindAttrGet(*self.handle.borrow(), value.to_c_api()),
                self,
            )
            .expect("invalid LLVM frame pointer kind attribute")
        }
    }

    /// Creates a new LLVM [`DiNullTypeAttributeRef`] owned by this [`Context`].
    pub fn llvm_di_null_type_attribute<'c>(&'c self) -> DiNullTypeAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        unsafe {
            DiNullTypeAttributeRef::from_c_api(mlirLLVMDINullTypeAttrGet(*self.handle.borrow()), self)
                .expect("invalid LLVM debug info null type attribute")
        }
    }

    /// Creates a new LLVM [`UndefAttributeRef`] owned by this [`Context`].
    pub fn llvm_undef_attribute<'c>(&'c self) -> UndefAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        unsafe {
            UndefAttributeRef::from_c_api(mlirLlvmUndefAttrGet(*self.handle.borrow()), self)
                .expect("invalid LLVM undef attribute")
        }
    }

    /// Creates a new LLVM [`PoisonAttributeRef`] owned by this [`Context`].
    pub fn llvm_poison_attribute<'c>(&'c self) -> PoisonAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        unsafe {
            PoisonAttributeRef::from_c_api(mlirLlvmPoisonAttrGet(*self.handle.borrow()), self)
                .expect("invalid LLVM poison attribute")
        }
    }

    /// Creates a new LLVM [`ZeroAttributeRef`] owned by this [`Context`].
    pub fn llvm_zero_attribute<'c>(&'c self) -> ZeroAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        unsafe {
            ZeroAttributeRef::from_c_api(mlirLlvmZeroAttrGet(*self.handle.borrow()), self)
                .expect("invalid LLVM zero attribute")
        }
    }

    /// Creates a new LLVM [`MdStringAttributeRef`] owned by this [`Context`].
    pub fn llvm_md_string_attribute<'c, 's, S: Into<StringRef<'s>>>(
        &'c self,
        value: S,
    ) -> MdStringAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        unsafe {
            MdStringAttributeRef::from_c_api(
                mlirLLVMMDStringAttrGet(*self.handle.borrow(), value.into().to_c_api()),
                self,
            )
            .expect("invalid LLVM metadata string attribute")
        }
    }

    /// Creates a new LLVM [`MdConstantAttributeRef`] owned by this [`Context`].
    pub fn llvm_md_constant_attribute<'c, A: Attribute<'c, 't>>(&'c self, value: A) -> MdConstantAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        unsafe {
            MdConstantAttributeRef::from_c_api(mlirLLVMMDConstantAttrGet(*self.handle.borrow(), value.to_c_api()), self)
                .expect("invalid LLVM metadata constant attribute")
        }
    }

    /// Creates a new LLVM [`MdFuncAttributeRef`] owned by this [`Context`].
    pub fn llvm_md_func_attribute<'c, 's, S: Into<StringRef<'s>>>(&'c self, name: S) -> MdFuncAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        let name = self.flat_symbol_ref_attribute(name);
        unsafe {
            MdFuncAttributeRef::from_c_api(mlirLLVMMDFuncAttrGet(*self.handle.borrow(), name.to_c_api()), self)
                .expect("invalid LLVM metadata function attribute")
        }
    }

    /// Creates a new LLVM [`MdNodeAttributeRef`] owned by this [`Context`].
    pub fn llvm_md_node_attribute<'c>(&'c self, operands: &[AttributeRef<'c, 't>]) -> MdNodeAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::llvm());
        let operands = operands.iter().map(|operand| unsafe { operand.to_c_api() }).collect::<Vec<_>>();
        unsafe {
            MdNodeAttributeRef::from_c_api(
                mlirLLVMMDNodeAttrGet(*self.handle.borrow(), operands.len().cast_signed(), operands.as_ptr()),
                self,
            )
            .expect("invalid LLVM metadata node attribute")
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_calling_convention_attribute() {
        let context = Context::new();
        let attribute = context.llvm_calling_convention_attribute(CallingConvention::Fast);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.dialect().namespace().unwrap(), "llvm");
        assert_eq!(attribute.value(), CallingConvention::Fast);
    }

    #[test]
    fn test_calling_convention_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.llvm_calling_convention_attribute(CallingConvention::Fast);
        let attribute_2 = context.llvm_calling_convention_attribute(CallingConvention::Fast);
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.llvm_calling_convention_attribute(CallingConvention::Cold);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.llvm_calling_convention_attribute(CallingConvention::Fast);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_calling_convention_attribute_display_and_debug() {
        let context = Context::new();
        test_attribute_display_and_debug(
            context.llvm_calling_convention_attribute(CallingConvention::Fast),
            "#llvm.cconv<fastcc>",
        );
    }

    #[test]
    fn test_calling_convention_attribute_casting() {
        let context = Context::new();
        test_attribute_casting(context.llvm_calling_convention_attribute(CallingConvention::Fast));
    }

    #[test]
    fn test_comdat_attribute() {
        let context = Context::new();
        let attribute = context.llvm_comdat_attribute(Comdat::NoDeduplicate);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.dialect().namespace().unwrap(), "llvm");
        assert_eq!(attribute.value(), Comdat::NoDeduplicate);
    }

    #[test]
    fn test_comdat_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.llvm_comdat_attribute(Comdat::NoDeduplicate);
        let attribute_2 = context.llvm_comdat_attribute(Comdat::NoDeduplicate);
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.llvm_comdat_attribute(Comdat::Any);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.llvm_comdat_attribute(Comdat::NoDeduplicate);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_comdat_attribute_display_and_debug() {
        let context = Context::new();
        test_attribute_display_and_debug(
            context.llvm_comdat_attribute(Comdat::NoDeduplicate),
            "#llvm<comdat nodeduplicate>",
        );
    }

    #[test]
    fn test_comdat_attribute_casting() {
        let context = Context::new();
        test_attribute_casting(context.llvm_comdat_attribute(Comdat::NoDeduplicate));
    }

    #[test]
    fn test_linkage_attribute() {
        let context = Context::new();
        let attribute = context.llvm_linkage_attribute(Linkage::Internal);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.dialect().namespace().unwrap(), "llvm");
        assert_eq!(attribute.value(), Linkage::Internal);
    }

    #[test]
    fn test_linkage_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.llvm_linkage_attribute(Linkage::Internal);
        let attribute_2 = context.llvm_linkage_attribute(Linkage::Internal);
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.llvm_linkage_attribute(Linkage::Private);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.llvm_linkage_attribute(Linkage::Internal);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_linkage_attribute_display_and_debug() {
        let context = Context::new();
        test_attribute_display_and_debug(context.llvm_linkage_attribute(Linkage::Internal), "#llvm.linkage<internal>");
    }

    #[test]
    fn test_linkage_attribute_casting() {
        let context = Context::new();
        test_attribute_casting(context.llvm_linkage_attribute(Linkage::Internal));
    }

    #[test]
    fn test_address_space_attribute() {
        let context = Context::new();
        let attribute = context.llvm_address_space_attribute(3);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.dialect().namespace().unwrap(), "llvm");
        assert_eq!(attribute.address_space(), 3);
    }

    #[test]
    fn test_address_space_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.llvm_address_space_attribute(3);
        let attribute_2 = context.llvm_address_space_attribute(3);
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.llvm_address_space_attribute(4);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.llvm_address_space_attribute(3);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_address_space_attribute_display_and_debug() {
        let context = Context::new();
        test_attribute_display_and_debug(context.llvm_address_space_attribute(3), "#llvm.address_space<3>");
    }

    #[test]
    fn test_address_space_attribute_casting() {
        let context = Context::new();
        test_attribute_casting(context.llvm_address_space_attribute(3));
    }

    #[test]
    fn test_frame_pointer_kind_attribute() {
        let context = Context::new();
        let attribute = context.llvm_frame_pointer_kind_attribute(FramePointerKind::NonLeaf);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.dialect().namespace().unwrap(), "llvm");
        assert_eq!(attribute.value(), FramePointerKind::NonLeaf);
    }

    #[test]
    fn test_frame_pointer_kind_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.llvm_frame_pointer_kind_attribute(FramePointerKind::NonLeaf);
        let attribute_2 = context.llvm_frame_pointer_kind_attribute(FramePointerKind::NonLeaf);
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.llvm_frame_pointer_kind_attribute(FramePointerKind::All);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.llvm_frame_pointer_kind_attribute(FramePointerKind::NonLeaf);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_frame_pointer_kind_attribute_display_and_debug() {
        let context = Context::new();
        test_attribute_display_and_debug(
            context.llvm_frame_pointer_kind_attribute(FramePointerKind::NonLeaf),
            "#llvm.framePointerKind<\"non-leaf\">",
        );
    }

    #[test]
    fn test_frame_pointer_kind_attribute_casting() {
        let context = Context::new();
        test_attribute_casting(context.llvm_frame_pointer_kind_attribute(FramePointerKind::NonLeaf));
    }

    #[test]
    fn test_di_null_type_attribute() {
        let context = Context::new();
        let attribute = context.llvm_di_null_type_attribute();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.dialect().namespace().unwrap(), "llvm");
    }

    #[test]
    fn test_di_null_type_attribute_display_and_debug() {
        let context = Context::new();
        test_attribute_display_and_debug(context.llvm_di_null_type_attribute(), "#llvm.di_null_type");
    }

    #[test]
    fn test_di_null_type_attribute_casting() {
        let context = Context::new();
        test_attribute_casting(context.llvm_di_null_type_attribute());
    }

    #[test]
    fn test_constant_like_attributes() {
        let context = Context::new();
        let undef = context.llvm_undef_attribute();
        let poison = context.llvm_poison_attribute();
        let zero = context.llvm_zero_attribute();
        assert_eq!(&context, undef.context());
        assert_eq!(&context, poison.context());
        assert_eq!(&context, zero.context());
        test_attribute_display_and_debug(undef, "#llvm.undef");
        test_attribute_display_and_debug(poison, "#llvm.poison");
        test_attribute_display_and_debug(zero, "#llvm.zero");
        test_attribute_casting(undef);
        test_attribute_casting(poison);
        test_attribute_casting(zero);
    }

    #[test]
    fn test_metadata_attributes() {
        let context = Context::new();
        let string = context.llvm_md_string_attribute("foo.buffer");
        let constant =
            context.llvm_md_constant_attribute(context.integer_attribute(context.signless_integer_type(32), 42));
        let function = context.llvm_md_func_attribute("callee");
        let node = context.llvm_md_node_attribute(&[constant.as_ref(), string.as_ref(), function.as_ref()]);
        assert_eq!(string.value().as_str().unwrap(), "foo.buffer");
        assert_eq!(constant.value().to_string(), "42 : i32");
        assert_eq!(function.name().reference().as_str().unwrap(), "callee");
        assert_eq!(node.operand_count(), 3);
        assert_eq!(
            node.operands().map(|operand| operand.to_string()).collect::<Vec<_>>(),
            vec!["#llvm.md_const<42 : i32>", "#llvm.md_string<\"foo.buffer\">", "#llvm.md_func<@callee>",]
        );
        test_attribute_display_and_debug(string, "#llvm.md_string<\"foo.buffer\">");
        test_attribute_display_and_debug(constant, "#llvm.md_const<42 : i32>");
        test_attribute_display_and_debug(function, "#llvm.md_func<@callee>");
        test_attribute_display_and_debug(
            node,
            "#llvm.md_node<#llvm.md_const<42 : i32>, #llvm.md_string<\"foo.buffer\">, #llvm.md_func<@callee>>",
        );
        test_attribute_casting(string);
        test_attribute_casting(constant);
        test_attribute_casting(function);
        test_attribute_casting(node);
    }
}
