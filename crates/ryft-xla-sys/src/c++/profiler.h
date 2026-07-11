#pragma once

#include <stddef.h>
#include <stdint.h>

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct RYFT_XLA_Profiler_XSpace_To_Profiled_Instructions_Args {
  const uint8_t *xspace;
  size_t xspace_size;
  uint8_t *profile;
  size_t profile_size;
  uint8_t *error;
  size_t error_size;
};

RYFT_XLA_SYS_EXPORT void RYFT_XLA_Profiler_XSpace_To_Profiled_Instructions(
    RYFT_XLA_Profiler_XSpace_To_Profiled_Instructions_Args *args);

struct RYFT_XLA_Profiler_Aggregate_Profiled_Instructions_Args {
  const uint8_t *const *profiles;
  const size_t *profile_sizes;
  size_t profile_count;
  int32_t percentile;
  uint8_t *profile;
  size_t profile_size;
  uint8_t *error;
  size_t error_size;
};

RYFT_XLA_SYS_EXPORT void RYFT_XLA_Profiler_Aggregate_Profiled_Instructions(
    RYFT_XLA_Profiler_Aggregate_Profiled_Instructions_Args *args);

RYFT_XLA_SYS_EXPORT void RYFT_XLA_Profiler_Byte_Buffer_Destroy(uint8_t *buffer);

#ifdef __cplusplus
}
#endif
