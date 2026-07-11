#include "profiler.h"

#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "xla/python/aggregate_profile.h"
#include "xla/python/xplane_to_profile_instructions.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace {

void CopyBytes(const std::string &source, uint8_t **destination, size_t *destination_size) {
  *destination_size = source.size();
  if (source.empty()) {
    *destination = nullptr;
    return;
  }
  *destination = new uint8_t[source.size()];
  std::memcpy(*destination, source.data(), source.size());
}

void SetError(const std::string &message, uint8_t **error, size_t *error_size) {
  CopyBytes(message, error, error_size);
}

}  // namespace

void RYFT_XLA_Profiler_XSpace_To_Profiled_Instructions(RYFT_XLA_Profiler_XSpace_To_Profiled_Instructions_Args *args) {
  if (args->xspace_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    SetError("xspace profile exceeds the Protobuf parser size limit", &args->error, &args->error_size);
    return;
  }
  tensorflow::profiler::XSpace xspace;
  if (!xspace.ParseFromArray(args->xspace, static_cast<int>(args->xspace_size))) {
    SetError("failed to parse XSpace profile", &args->error, &args->error_size);
    return;
  }
  tensorflow::profiler::ProfiledInstructionsProto profile;
  const auto status = xla::ConvertXplaneToProfiledInstructionsProto({std::move(xspace)}, &profile);
  if (!status.ok()) {
    SetError(status.ToString(), &args->error, &args->error_size);
    return;
  }
  CopyBytes(profile.SerializeAsString(), &args->profile, &args->profile_size);
}

void RYFT_XLA_Profiler_Aggregate_Profiled_Instructions(RYFT_XLA_Profiler_Aggregate_Profiled_Instructions_Args *args) {
  std::vector<tensorflow::profiler::ProfiledInstructionsProto> profiles;
  profiles.reserve(args->profile_count);
  for (size_t index = 0; index < args->profile_count; ++index) {
    if (args->profile_sizes[index] > static_cast<size_t>(std::numeric_limits<int>::max())) {
      SetError("feedback-directed optimization profile exceeds the Protobuf parser size limit", &args->error,
               &args->error_size);
      return;
    }
    tensorflow::profiler::ProfiledInstructionsProto profile;
    if (!profile.ParseFromArray(args->profiles[index], static_cast<int>(args->profile_sizes[index]))) {
      SetError("failed to parse feedback-directed optimization profile", &args->error, &args->error_size);
      return;
    }
    profiles.push_back(std::move(profile));
  }
  tensorflow::profiler::ProfiledInstructionsProto aggregate;
  xla::AggregateProfiledInstructionsProto(profiles, args->percentile, &aggregate);
  CopyBytes(aggregate.SerializeAsString(), &args->profile, &args->profile_size);
}

void RYFT_XLA_Profiler_Byte_Buffer_Destroy(uint8_t *buffer) { delete[] buffer; }
