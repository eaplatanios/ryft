#include "bufferization.h"

#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Bufferization, bufferization,
                                      mlir::bufferization::BufferizationDialect)
