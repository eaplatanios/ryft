#include "triton_gpu.h"

#if defined(RYFT_TRITON_COMPILER)
#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TritonTtg, ttg, mlir::triton::gpu::TritonGPUDialect)

MlirAttribute mlirTritonTtgBarrierAddressSpacesAttrGet(MlirContext context, uint32_t spaces) {
  auto value = mlir::triton::gpu::symbolizeAddrSpace(spaces);
  if (context.ptr == nullptr || !value) return {nullptr};
  return wrap(mlir::triton::gpu::AddrSpaceAttr::get(unwrap(context), *value));
}

MlirAttribute mlirTritonTtgBarrierScopeAttrGet(MlirContext context, uint32_t scope) {
  auto value = mlir::triton::gpu::symbolizeBarrierScope(scope);
  if (context.ptr == nullptr || !value) return {nullptr};
  return wrap(mlir::triton::gpu::BarrierScopeAttr::get(unwrap(context), *value));
}

MlirType mlirTritonTtgAsyncTokenTypeGet(MlirContext context) {
  if (context.ptr == nullptr) return {nullptr};
  return wrap(mlir::triton::gpu::AsyncTokenType::get(unwrap(context)));
}

MlirType mlirTritonTtgMutableSharedMemDescTypeGet(MlirType elementType, intptr_t rank, const int64_t* shape) {
  if (elementType.ptr == nullptr || shape == nullptr || rank <= 0 || rank > std::numeric_limits<int>::max()) {
    return {nullptr};
  }
  llvm::ArrayRef<int64_t> dimensions(shape, static_cast<size_t>(rank));
  if (std::any_of(dimensions.begin(), dimensions.end(), [](int64_t size) { return size <= 0; })) {
    return {nullptr};
  }
  auto element = unwrap(elementType);
  auto* context = element.getContext();
  std::vector<unsigned> order(static_cast<size_t>(rank));
  std::iota(order.rbegin(), order.rend(), 0u);
  auto ctaLayout = mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(context, static_cast<int>(rank));
  auto layout = mlir::triton::gpu::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, order, ctaLayout);
  auto space = mlir::triton::gpu::SharedMemorySpaceAttr::get(context);
  return wrap(mlir::triton::gpu::MemDescType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(context)); }, dimensions, element, layout, space, true));
}
#else
MlirDialectHandle mlirGetDialectHandle__ttg__(void) { return {nullptr}; }
MlirAttribute mlirTritonTtgBarrierAddressSpacesAttrGet(MlirContext, uint32_t) { return {nullptr}; }
MlirAttribute mlirTritonTtgBarrierScopeAttrGet(MlirContext, uint32_t) { return {nullptr}; }
MlirType mlirTritonTtgAsyncTokenTypeGet(MlirContext) { return {nullptr}; }
MlirType mlirTritonTtgMutableSharedMemDescTypeGet(MlirType, intptr_t, const int64_t*) { return {nullptr}; }
#endif
