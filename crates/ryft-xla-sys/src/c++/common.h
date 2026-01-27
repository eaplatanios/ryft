#pragma once

// We only include the correct C++ types when compiling using a C++ compiler.
// Otherwise, we use placeholders that act as opaque type pointers for `bindgen` to use.

#ifdef __cplusplus
#include <memory>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/distributed/distributed.h"
extern "C" {
#endif

#ifdef __cplusplus
typedef xla::DistributedRuntimeService PJRT_Distributed_Runtime_Service;
typedef xla::DistributedRuntimeClient PJRT_Distributed_Runtime_Client;
#else
typedef struct _DistributedRuntimeService PJRT_Distributed_Runtime_Service;
typedef struct _DistributedRuntimeClient PJRT_Distributed_Runtime_Client;
#endif

struct PJRT_Error {
  absl::Status status;
};

#ifdef __cplusplus
}
#endif

// We only include our macros when compiling using a C++ compiler.
// They are ignored when using `bindgen`.

#ifdef __cplusplus

#define PJRT_RETURN_IF_ERROR(expr)                                \
  do {                                                            \
    absl::Status _status = (expr);                                \
    if (!_status.ok()) {                                          \
      PJRT_Error* _c_status = new PJRT_Error{std::move(_status)}; \
      return _c_status;                                           \
    }                                                             \
  } while (false)

#define PJRT_ASSIGN_OR_RETURN(lhs, rexpr)                                  \
  _PJRT_ASSIGN_OR_RETURN_IMPL(_PJRT_CONCAT(_status_or_value, __COUNTER__), \
                              lhs, rexpr,                                  \
                              _PJRT_CONCAT(_c_status, __COUNTER__));

#define _PJRT_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr, c_status) \
  auto statusor = (rexpr);                                          \
  if (!statusor.ok()) {                                             \
    PJRT_Error* c_status = new PJRT_Error();                        \
    c_status->status = statusor.status();                           \
    return c_status;                                                \
  }                                                                 \
  lhs = std::move(*statusor)

#define _PJRT_CONCAT(x, y) _PJRT_CONCAT_IMPL(x, y)
#define _PJRT_CONCAT_IMPL(x, y) x##y

#endif
