#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "common.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrows a module for synchronous compilation. The caller must hold its context exclusively for the entire call.
// Compilation clones the module before lowering. All output buffers are owned by these arguments and must be destroyed
// after either success or failure. Initialize output fields to zero and do not reuse arguments before destroying them.
typedef struct RYFT_XLA_Triton_Compile_Args {
  MlirModule module;
  MlirStringRef platform;
  MlirStringRef architecture;
  int32_t warp_count;
  int32_t stage_count;
  size_t maximum_artifact_bytes;
  size_t maximum_diagnostic_bytes;
  uint8_t *artifact;
  size_t artifact_size;
  uint8_t *entry_name;
  size_t entry_name_size;
  uint8_t *diagnostics;
  size_t diagnostics_size;
  int64_t argument_count;
  int64_t actual_warp_count;
  int64_t threads_per_warp;
  int64_t shared_memory_bytes;
} RYFT_XLA_Triton_Compile_Args;

// Returns failure with diagnostics when the target is unavailable or compilation fails. Native aborts are not caught.
RYFT_XLA_SYS_EXPORT MlirLogicalResult RYFT_XLA_Triton_Compile(RYFT_XLA_Triton_Compile_Args *args);

// Releases every owned output buffer and resets output fields. Does not destroy the borrowed module or context.
RYFT_XLA_SYS_EXPORT void RYFT_XLA_Triton_Compile_Args_Destroy(RYFT_XLA_Triton_Compile_Args *args);

// Source revisions borrow process-lifetime storage. The assembler version is owned and must be destroyed after query.
typedef struct RYFT_XLA_Triton_Versions {
  MlirStringRef xla;
  MlirStringRef jax;
  MlirStringRef triton;
  MlirStringRef rocm_device_libs;
  bool cuda_available;
  bool rocm_available;
  int32_t cuda_toolkit_version;
  uint8_t *assembler_version;
  size_t assembler_version_size;
} RYFT_XLA_Triton_Versions;

// Queries the linked compiler and effective CUDA assembler without initializing a GPU context.
RYFT_XLA_SYS_EXPORT RYFT_XLA_Triton_Versions RYFT_XLA_Triton_Get_Versions(void);

// Releases the owned assembler version and resets the version fields.
RYFT_XLA_SYS_EXPORT void RYFT_XLA_Triton_Versions_Destroy(RYFT_XLA_Triton_Versions *versions);

#ifdef __cplusplus
}
#endif
