#include "triton_compiler.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>

#if defined(RYFT_TRITON_COMPILER)
#if defined(RYFT_TRITON_CUDA)
#include "cuda.h"
#endif
#include "llvm/ADT/BitVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/debug_options_flags.h"
#include "xla/service/gpu/llvm_gpu_backend/amdgpu_backend.h"
#include "xla/service/gpu/target_constants.h"
#if defined(RYFT_TRITON_CUDA)
#include "xla/service/gpu/llvm_gpu_backend/nvptx_backend.h"
#include "xla/stream_executor/cuda/subprocess_compilation.h"
#endif
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

#endif

namespace {

// Copies a native product into an owned buffer with the same allocator used by the public destroy functions.
void CopyBytes(const std::string& source, uint8_t** destination, size_t* destination_size) {
  *destination_size = source.size();
  *destination = source.empty() ? nullptr : new uint8_t[source.size()];
  if (!source.empty()) std::memcpy(*destination, source.data(), source.size());
}

#if defined(RYFT_TRITON_COMPILER)

// Bounds captured diagnostics without redirecting process-global streams or allowing a diagnostic to grow unbounded.
class BoundedDiagnostics final : public llvm::raw_ostream {
 public:
  // Constructs an unbuffered capture with an explicit byte limit.
  explicit BoundedDiagnostics(size_t limit) : limit_(limit) { SetUnbuffered(); }
  // Returns the captured diagnostic prefix.
  const std::string& content() const { return content_; }
  // Reports whether any diagnostic bytes exceeded the limit.
  bool exceeded() const { return exceeded_; }

 private:
  // Appends only the available diagnostic bytes while retaining overflow as a compilation failure.
  void write_impl(const char* data, size_t size) override {
    size_t available = limit_ - content_.size();
    exceeded_ |= size > available;
    content_.append(data, std::min(size, available));
  }
  // Reports the number of diagnostic bytes retained by the stream.
  uint64_t current_pos() const override { return content_.size(); }
  size_t limit_;
  std::string content_;
  bool exceeded_ = false;
};

// Parses a complete positive decimal option, without accepting trailing characters or integer overflow.
bool ParseInteger(const std::string& text, int& value) {
  auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc() && result.ptr == text.data() + text.size() && value > 0;
}

// Compiles a clone of one static pointer-only TTIR entry. Unsupported ABI extensions fail before publication.
bool Compile(RYFT_XLA_Triton_Compile_Args* args, llvm::raw_ostream& diagnostics) {
  std::string platform(unwrap(args->platform));
  if (platform != "cuda" && platform != "rocm") {
    diagnostics << "expected `cuda` or `rocm` platform";
    return false;
  }
  bool cuda = platform == "cuda";
#if !defined(RYFT_TRITON_CUDA)
  if (cuda) {
    diagnostics << "`cuda` compilation is not included in this compiler\n";
    return false;
  }
#endif
  std::string architecture(unwrap(args->architecture));
  int major = 0, minor = 0, warps = args->warp_count, stages = args->stage_count;
  bool valid_architecture = false;
  if (cuda) {
    auto separator = architecture.find('.');
    // CUDA minor zero is accepted separately because ordinary options must be positive.
    std::string minor_text = separator == std::string::npos ? "" : architecture.substr(separator + 1);
    valid_architecture = separator != std::string::npos && ParseInteger(architecture.substr(0, separator), major) &&
                         (minor_text == "0" || ParseInteger(minor_text, minor)) && major >= 8 && major <= 12 &&
                         minor <= 9;
  } else {
    valid_architecture = stream_executor::RocmComputeCapability(architecture).is_supported_gfx_version() &&
                         architecture.find(':') == std::string::npos;
    for (const char* variable :
         {"TF_XLA_HSACO_CACHE_DIR", "TF_XLA_HSACO_BITCODE_SIZE_THRESHOLD", "TF_ROCM_KEEP_XLA_TEMPFILES"}) {
      if (std::getenv(variable) != nullptr) {
        diagnostics << "unsupported inherited HSACO cache configuration\n";
        return false;
      }
    }
  }
  if (!valid_architecture || (warps != 1 && warps != 2 && warps != 4 && warps != 8) || stages < 1 || stages > 8) {
    diagnostics << "invalid target or compiler options\n";
    return false;
  }
  if (!args->module.ptr) {
    diagnostics << "expected a non-null Triton module";
    return false;
  }
  auto original = unwrap(args->module);
  auto* context = original.getContext();
  // The caller holds this context exclusively. The handler exists only for this synchronous compilation, while the
  // cloned module preserves every operation and attribute in the caller's original module.
  mlir::ScopedDiagnosticHandler handler(context, [&](mlir::Diagnostic& diagnostic) {
    diagnostic.print(diagnostics);
    diagnostics << "\n";
    return mlir::success();
  });
  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::LLVM::LLVMDialect>();
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  context->appendDialectRegistry(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module(original.clone());
  if (!module || mlir::failed(mlir::verify(*module))) return false;
  auto functions = module->getOps<mlir::triton::FuncOp>();
  if (std::distance(functions.begin(), functions.end()) != 1) {
    diagnostics << "expected exactly one Triton function\n";
    return false;
  }
  auto function = *functions.begin();
  if (function.isExternal() || function.isPrivate() || function.getFunctionType().getNumResults() != 0) {
    diagnostics << "expected a public defined entry without results\n";
    return false;
  }
  for (auto argument : function.getArgumentTypes()) {
    auto pointer = mlir::dyn_cast<mlir::triton::PointerType>(argument);
    if (!pointer || static_cast<int>(pointer.getAddressSpace()) != 1) {
      diagnostics << "expected pointer-only entry arguments\n";
      return false;
    }
  }
  std::string entry = function.getName().str();
  int64_t argument_count = function.getNumArguments();
  stream_executor::GpuComputeCapability capability = stream_executor::CudaComputeCapability(major, minor);
  int threads_per_warp = 32;
  if (!cuda) {
    stream_executor::RocmComputeCapability rocm_capability(architecture);
    threads_per_warp = rocm_capability.threads_per_warp();
    capability = rocm_capability;
  }
  mlir::PassManager passes(context);
  passes.enableVerifier(true);
  xla::gpu::CreateTritonPipeline(&passes, capability, warps, 1, stages);
  if (mlir::failed(passes.run(*module))) return false;
  auto shared = (*module)->getAttrOfType<mlir::IntegerAttr>("ttg.shared");
  auto ctas = (*module)->getAttrOfType<mlir::IntegerAttr>("ttg.num-ctas");
  auto profile_scratch = (*module)->getAttrOfType<mlir::IntegerAttr>("ttg.profile_scratch_memory_size");
  auto total_warps = (*module)->getAttrOfType<mlir::IntegerAttr>("ttg.total-num-warps");
  int64_t actual_warps = total_warps ? total_warps.getInt() : warps;
  auto scratch = (*module)->getAttrOfType<mlir::IntegerAttr>("ttg.global_scratch_memory_size");
  if (!shared || shared.getInt() < 0 || !ctas || ctas.getInt() != 1 ||
      (profile_scratch && profile_scratch.getInt() != 0) || actual_warps <= 0 ||
      actual_warps > 1024 / threads_per_warp || (scratch && scratch.getInt() != 0)) {
    diagnostics << "unsupported compiler resource metadata\n";
    return false;
  }
  auto lowered_entry = module->lookupSymbol<mlir::LLVM::LLVMFuncOp>(entry);
  if (!lowered_entry || lowered_entry.getNumArguments() != argument_count + 2 ||
      !lowered_entry.getArgument(argument_count).use_empty() ||
      !lowered_entry.getArgument(argument_count + 1).use_empty()) {
    diagnostics << "unsupported compiler scratch arguments\n";
    return false;
  }
  // Pinned Triton appends global scratch and profiling pointers even when unused. Only those two proven unused
  // parameters may be removed; eraseArguments updates the function type, argument attributes, and entry block.
  llvm::BitVector erased_arguments(argument_count + 2);
  erased_arguments.set(argument_count);
  erased_arguments.set(argument_count + 1);
  if (mlir::failed(lowered_entry.eraseArguments(erased_arguments))) return false;
  // Triton preserves pointee element types for its optional LLVM debug-info passes. This pipeline does not run those
  // passes, and the LLVM translator cannot consume this TT-only attribute. The pointer ABI is checked independently.
  for (unsigned index = 0; index < lowered_entry.getNumArguments(); ++index) {
    lowered_entry.removeArgAttr(index, "tt.pointee_type");
  }
  if (mlir::failed(mlir::verify(*module))) return false;
  llvm::LLVMContext llvm_context;
  auto llvm_module = mlir::translateModuleToLLVMIR(*module, llvm_context);
  if (!llvm_module) return false;
  // The admitted subset must not resolve device math libraries from an ambient toolkit installation.
  for (auto iterator = llvm_module->begin(); iterator != llvm_module->end();) {
    auto& candidate = *iterator++;
    if (!candidate.isDeclaration() || candidate.isIntrinsic()) continue;
    if (candidate.use_empty()) {
      candidate.eraseFromParent();
      continue;
    }
    // CUDA provides this exact assertion builtin in the device runtime; it does not require ambient libdevice.
    bool assertion = cuda && candidate.getName() == "__assertfail" && !candidate.isVarArg() &&
                     candidate.getReturnType()->isVoidTy() && candidate.arg_size() == 5 && candidate.doesNotReturn();
    if (assertion) {
      for (unsigned index : {0, 1, 3}) {
        auto* pointer = candidate.getArg(index)->getType();
        assertion &= pointer->isPointerTy() && pointer->getPointerAddressSpace() == 0;
      }
      assertion &= candidate.getArg(2)->getType()->isIntegerTy(32) && candidate.getArg(4)->getType()->isIntegerTy(64);
    }
    if (!assertion) {
      diagnostics << "unsupported external device function\n";
      return false;
    }
  }
  auto* llvm_function = llvm_module->getFunction(entry);
  if (!llvm_function || llvm_function->arg_size() != argument_count || llvm_function->isVarArg() ||
      !llvm_function->getReturnType()->isVoidTy()) {
    diagnostics << "compiler changed the entry argument ABI\n";
    return false;
  }
  for (const auto& argument : llvm_function->args()) {
    if (!argument.getType()->isPointerTy() || argument.getType()->getPointerAddressSpace() != 1) {
      diagnostics << "compiler changed pointer address spaces\n";
      return false;
    }
  }
  std::string artifact;
  if (cuda) {
#if defined(RYFT_TRITON_CUDA)
    auto compiled = xla::gpu::nvptx::CompileToPtx(
        llvm_module.get(), capability, xla::DefaultDebugOptionsIgnoringFlags(), [&](llvm::TargetMachine* target) {
          llvm_module->setDataLayout(target->createDataLayout());
          llvm_module->setTargetTriple(target->getTargetTriple());
        });
    if (!compiled.ok()) {
      diagnostics << compiled.status().ToString() << "\n";
      return false;
    }
    artifact = std::move(*compiled);
#endif
  } else {
    // The direct AMD path must mark the entry as a kernel after translation; XLA's shared pipeline leaves this
    // to its fusion compiler. These fixed launch attributes follow the pinned native Triton backend contract.
    llvm_module->setTargetTriple(llvm::Triple(xla::gpu::amdgpu::TargetTriple()));
    llvm_function->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    llvm_function->addFnAttr("amdgpu-cluster-dims", "1,1,1");
    llvm_function->addFnAttr("amdgpu-flat-work-group-size", "1," + std::to_string(actual_warps * threads_per_warp));
    llvm_function->addFnAttr("uniform-work-group-size", "true");
    llvm_function->addFnAttr("denormal-fp-math-f32", "ieee");
    // The pinned compiler hashes LLVM bitcode, target, and options itself. Its obsolete cache-key argument is ignored;
    // inherited persistent-cache settings were rejected above. Its process-wide in-memory cache remains available.
    auto compiled =
        xla::gpu::amdgpu::CompileToHsaco(llvm_module.get(), capability, xla::DefaultDebugOptionsIgnoringFlags(), "");
    if (!compiled.ok()) {
      diagnostics << compiled.status().ToString() << "\n";
      return false;
    }
    artifact.assign(reinterpret_cast<const char*>(compiled->hsaco.data()), compiled->hsaco.size());
  }
  if (artifact.size() > args->maximum_artifact_bytes) {
    diagnostics << "compiler artifact exceeds its byte limit";
    return false;
  }
  CopyBytes(artifact, &args->artifact, &args->artifact_size);
  CopyBytes(entry, &args->entry_name, &args->entry_name_size);
  args->argument_count = argument_count;
  args->actual_warp_count = actual_warps;
  args->threads_per_warp = threads_per_warp;
  args->shared_memory_bytes = shared.getInt();
  return true;
}
#endif
}  // namespace

MlirLogicalResult RYFT_XLA_Triton_Compile(RYFT_XLA_Triton_Compile_Args* args) {
#if defined(RYFT_TRITON_COMPILER)
  BoundedDiagnostics diagnostics(args->maximum_diagnostic_bytes);
  bool success = Compile(args, diagnostics) && !diagnostics.exceeded();
  if (!success) RYFT_XLA_Triton_Compile_Args_Destroy(args);
  CopyBytes(diagnostics.content(), &args->diagnostics, &args->diagnostics_size);
  return {static_cast<int8_t>(success)};
#else
  std::string diagnostics = "triton compilation is unavailable in this native archive";
  diagnostics.resize(std::min(diagnostics.size(), args->maximum_diagnostic_bytes));
  CopyBytes(diagnostics, &args->diagnostics, &args->diagnostics_size);
  return {0};
#endif
}

void RYFT_XLA_Triton_Compile_Args_Destroy(RYFT_XLA_Triton_Compile_Args* args) {
  delete[] args->artifact;
  delete[] args->entry_name;
  delete[] args->diagnostics;
  args->artifact = nullptr;
  args->artifact_size = 0;
  args->entry_name = nullptr;
  args->entry_name_size = 0;
  args->diagnostics = nullptr;
  args->diagnostics_size = 0;
  args->argument_count = 0;
  args->actual_warp_count = 0;
  args->threads_per_warp = 0;
  args->shared_memory_bytes = 0;
}

RYFT_XLA_Triton_Versions RYFT_XLA_Triton_Get_Versions() {
  RYFT_XLA_Triton_Versions versions{};
  versions.xla = mlirStringRefCreateFromCString("eb6b90ed013f511eca088c52f541f3c0819f919e");
  versions.jax = mlirStringRefCreateFromCString("a7606f995e1a92707cbeb257e487fa53e7abe84b");
  versions.triton = mlirStringRefCreateFromCString("a77e7c793abc0d0c923a9afb275058e2fe57a198");
  versions.rocm_device_libs = mlirStringRefCreateFromCString("53996464fa8d94b182ac4aaa7dc3a109ab524f45");
  std::string assembler_version = "unavailable";
#if defined(RYFT_TRITON_COMPILER)
  versions.rocm_available = true;
#endif
#if defined(RYFT_TRITON_CUDA)
  versions.cuda_available = true;
  versions.cuda_toolkit_version = CUDA_VERSION;
  auto assembler =
      stream_executor::GetAsmCompilerVersion(xla::DefaultDebugOptionsIgnoringFlags().xla_gpu_cuda_data_dir());
  if (assembler.ok()) assembler_version = assembler->ToString();
#endif
  CopyBytes(assembler_version, &versions.assembler_version, &versions.assembler_version_size);
  return versions;
}

void RYFT_XLA_Triton_Versions_Destroy(RYFT_XLA_Triton_Versions* versions) {
  delete[] versions->assembler_version;
  *versions = {};
}
