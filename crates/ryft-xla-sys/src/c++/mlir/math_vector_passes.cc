#include "mlir/CAPI/Pass.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

using namespace mlir::arith;

#define mlirRegisterPasses mlirRegisterArithPasses
#define registerPasses registerArithPasses
extern "C" {
#include "mlir/Dialect/Arith/Transforms/Passes.capi.cpp.inc"
}
#undef mlirRegisterPasses
#undef registerPasses

using namespace mlir::LLVM;

#define mlirRegisterPasses mlirRegisterLLVMPasses
#define registerPasses registerLLVMPasses
extern "C" {
#include "mlir/Dialect/LLVMIR/Transforms/Passes.capi.cpp.inc"
}
#undef mlirRegisterPasses
#undef registerPasses

using namespace mlir::math;

#define mlirRegisterPasses mlirRegisterMathPasses
#define registerPasses registerMathPasses
extern "C" {
#include "mlir/Dialect/Math/Transforms/Passes.capi.cpp.inc"
}
#undef mlirRegisterPasses
#undef registerPasses

using namespace mlir::memref;

#define mlirRegisterPasses mlirRegisterMemRefPasses
#define registerPasses registerMemRefPasses
extern "C" {
#include "mlir/Dialect/MemRef/Transforms/Passes.capi.cpp.inc"
}
#undef mlirRegisterPasses
#undef registerPasses

using namespace mlir::vector;

#define mlirRegisterPasses mlirRegisterVectorPasses
#define registerPasses registerVectorPasses
extern "C" {
#include "mlir/Dialect/Vector/Transforms/Passes.capi.cpp.inc"
}
#undef mlirRegisterPasses
#undef registerPasses
