#include "builtin.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinTypes.h"

bool mlirTypeIsAToken(MlirType type) {
  return type.ptr != nullptr && llvm::isa<mlir::TokenType>(unwrap(type));
}

MlirType mlirTokenTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::TokenType::get(unwrap(context)));
}
