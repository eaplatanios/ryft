#include "xla/tsl/platform/grpc_credentials.h"

#include "distributed.h"

const PJRT_Error* PJRT_Distributed_Runtime_Service_New(
  PJRT_Distributed_Runtime_Service_New_Args* args) {
  xla::CoordinationServiceImpl::Options options;
  options.num_nodes = args->num_nodes;
  options.heartbeat_timeout = absl::Seconds(args->heartbeat_timeout);
  options.cluster_register_timeout = absl::Seconds(args->cluster_register_timeout);
  options.shutdown_timeout = absl::Seconds(args->shutdown_timeout);
  PJRT_ASSIGN_OR_RETURN(
    std::unique_ptr<xla::DistributedRuntimeService> service,
    GetDistributedRuntimeService(std::string(args->address), options)
  );
  args->service = service.release();
  return new PJRT_Error{absl::Status()};
}

void PJRT_Distributed_Runtime_Service_Shutdown(
  PJRT_Distributed_Runtime_Service_Shutdown_Args* args) {
  args->service->Shutdown();
}

void PJRT_Distributed_Runtime_Service_Destroy(
  PJRT_Distributed_Runtime_Service_Destroy_Args* args) {
  delete args->service;
}

const PJRT_Error* PJRT_Distributed_Runtime_Client_New(
  PJRT_Distributed_Runtime_Client_New_Args* args) {
  xla::DistributedRuntimeClient::Options options;
  options.node_id = args->node_id;
  options.rpc_timeout = absl::Seconds(args->rpc_timeout);
  options.init_timeout = absl::Seconds(args->init_timeout);
  options.shutdown_timeout = absl::Seconds(args->shutdown_timeout);
  options.heartbeat_timeout = absl::Seconds(args->heartbeat_timeout);
  options.missed_heartbeat_callback = [
    user_arg = args->missed_heartbeat_callback_user_arg,
    callback = args->missed_heartbeat_callback
  ](absl::Status status) {
    auto error = new PJRT_Error{status};
    callback(error, user_arg);
  };
  options.shutdown_on_destruction = args->shutdown_on_destruction;
  options.recoverable = args->recoverable;
  auto channel = xla::GetDistributedRuntimeClientChannel(
    std::string(args->address), tsl::GetClientCredentials(false), args->use_compression);
  args->client = GetDistributedRuntimeClient(channel, options).release();
  return new PJRT_Error{absl::Status()};
}

PJRT_Error* PJRT_Distributed_Runtime_Client_Connect(
  PJRT_Distributed_Runtime_Client_Connect_Args* args) {
  return new PJRT_Error{args->client->Connect()};
}

PJRT_Error* PJRT_Distributed_Runtime_Client_Blocking_Key_Value_Get(
  PJRT_Distributed_Runtime_Client_Blocking_Key_Value_Get_Args* args) {
  PJRT_ASSIGN_OR_RETURN(
    auto _value,
    args->client->BlockingKeyValueGet(std::string(args->key), absl::Seconds(args->timeout))
  );
  char* value = new char[_value.size() + 1];
  strcpy(value, _value.c_str());
  args->value = value;
  return new PJRT_Error{absl::Status()};
}

PJRT_Error* PJRT_Distributed_Runtime_Client_Key_Value_Try_Get(
  PJRT_Distributed_Runtime_Client_Key_Value_Try_Get_Args* args) {
  PJRT_ASSIGN_OR_RETURN(
    auto _value,
    args->client->KeyValueTryGet(std::string(args->key))
  );
  char* value = new char[_value.size() + 1];
  strcpy(value, _value.c_str());
  args->value = value;
  return new PJRT_Error{absl::Status()};
}

PJRT_Error* PJRT_Distributed_Runtime_Client_Key_Value_Set(
  PJRT_Distributed_Runtime_Client_Key_Value_Set_Args* args) {
  return new PJRT_Error{
    args->client->KeyValueSet(std::string(args->key), std::string(args->value))
  };
}

PJRT_Error* PJRT_Distributed_Runtime_Client_Shutdown(
  PJRT_Distributed_Runtime_Client_Shutdown_Args* args) {
  return new PJRT_Error{args->client->Shutdown()};
}

void PJRT_Distributed_Runtime_Client_Destroy(
  PJRT_Distributed_Runtime_Client_Destroy_Args* args) {
  delete args->client;
}
