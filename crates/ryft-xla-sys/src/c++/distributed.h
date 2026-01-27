#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct PJRT_Distributed_Runtime_Service_New_Args {
  const char* address;
  uint32_t num_nodes;
  uint32_t heartbeat_timeout;
  uint32_t cluster_register_timeout;
  uint32_t shutdown_timeout;
  PJRT_Distributed_Runtime_Service* service;
};

const PJRT_Error* PJRT_Distributed_Runtime_Service_New(
  PJRT_Distributed_Runtime_Service_New_Args* args);

struct PJRT_Distributed_Runtime_Service_Shutdown_Args {
  PJRT_Distributed_Runtime_Service* service;
};

void PJRT_Distributed_Runtime_Service_Shutdown(
  PJRT_Distributed_Runtime_Service_Shutdown_Args* args);

struct PJRT_Distributed_Runtime_Service_Destroy_Args {
  PJRT_Distributed_Runtime_Service* service;
};

void PJRT_Distributed_Runtime_Service_Destroy(
  PJRT_Distributed_Runtime_Service_Destroy_Args* args);

typedef void (*PJRT_Distributed_Missed_Heartbeat_Callback)(
  PJRT_Error* error, void* user_arg);

struct PJRT_Distributed_Runtime_Client_New_Args {
  const char* address;
  uint32_t node_id;
  uint32_t rpc_timeout;
  uint32_t init_timeout;
  uint32_t shutdown_timeout;
  uint32_t heartbeat_timeout;
  PJRT_Distributed_Missed_Heartbeat_Callback missed_heartbeat_callback;
  void* missed_heartbeat_callback_user_arg;
  bool shutdown_on_destruction;
  bool recoverable;
  bool use_compression;
  PJRT_Distributed_Runtime_Client* client;
};

const PJRT_Error* PJRT_Distributed_Runtime_Client_New(
  PJRT_Distributed_Runtime_Client_New_Args* args);

struct PJRT_Distributed_Runtime_Client_Connect_Args {
  PJRT_Distributed_Runtime_Client* client;
};

PJRT_Error* PJRT_Distributed_Runtime_Client_Connect(
  PJRT_Distributed_Runtime_Client_Connect_Args* args);

struct PJRT_Distributed_Runtime_Client_Blocking_Key_Value_Get_Args {
  PJRT_Distributed_Runtime_Client* client;
  const char* key;
  uint32_t timeout;
  const char* value;
};

PJRT_Error* PJRT_Distributed_Runtime_Client_Blocking_Key_Value_Get(
  PJRT_Distributed_Runtime_Client_Blocking_Key_Value_Get_Args* args);

struct PJRT_Distributed_Runtime_Client_Key_Value_Try_Get_Args {
  PJRT_Distributed_Runtime_Client* client;
  const char* key;
  const char* value;
};

PJRT_Error* PJRT_Distributed_Runtime_Client_Key_Value_Try_Get(
  PJRT_Distributed_Runtime_Client_Key_Value_Try_Get_Args* args);

struct PJRT_Distributed_Runtime_Client_Key_Value_Set_Args {
  PJRT_Distributed_Runtime_Client* client;
  const char* key;
  const char* value;
};

PJRT_Error* PJRT_Distributed_Runtime_Client_Key_Value_Set(
  PJRT_Distributed_Runtime_Client_Key_Value_Set_Args* args);

struct PJRT_Distributed_Runtime_Client_Shutdown_Args {
  PJRT_Distributed_Runtime_Client* client;
};

PJRT_Error* PJRT_Distributed_Runtime_Client_Shutdown(
  PJRT_Distributed_Runtime_Client_Shutdown_Args* args);

struct PJRT_Distributed_Runtime_Client_Destroy_Args {
  PJRT_Distributed_Runtime_Client* client;
};

void PJRT_Distributed_Runtime_Client_Destroy(
  PJRT_Distributed_Runtime_Client_Destroy_Args* args);

#ifdef __cplusplus
}
#endif
