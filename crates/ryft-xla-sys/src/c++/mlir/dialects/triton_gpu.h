#pragma once

#include <stdint.h>

#include "../../common.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns a null handle when the native archive does not include the TritonGPU dialect.
RYFT_XLA_SYS_EXPORT MlirDialectHandle mlirGetDialectHandle__ttg__(void);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirTritonTtgBarrierAddressSpacesAttrGet(MlirContext context, uint32_t spaces);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirTritonTtgBarrierScopeAttrGet(MlirContext context, uint32_t scope);
RYFT_XLA_SYS_EXPORT MlirType mlirTritonTtgAsyncTokenTypeGet(MlirContext context);
RYFT_XLA_SYS_EXPORT MlirType mlirTritonTtgMutableSharedMemDescTypeGet(
    MlirType elementType, intptr_t rank, const int64_t* shape);

#ifdef __cplusplus
}
#endif
