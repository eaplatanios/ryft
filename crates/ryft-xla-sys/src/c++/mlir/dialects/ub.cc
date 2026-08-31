#include "ub.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(UB, ub, mlir::ub::UBDialect)

bool mlirAttributeIsAUbPoisonAttr(MlirAttribute attribute) {
  return attribute.ptr != nullptr && llvm::isa<mlir::ub::PoisonAttr>(unwrap(attribute));
}

MlirAttribute mlirUbPoisonAttrGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::ub::PoisonAttr::get(unwrap(context)));
}

MlirTypeID mlirUbPoisonAttrGetTypeID(void) {
  return wrap(mlir::ub::PoisonAttr::getTypeID());
}
