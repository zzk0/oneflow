/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_REQUEST_STORE_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_REQUEST_STORE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/collective_boxing/runtime_request_info.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

namespace boxing {

namespace collective {

class RequestEntry final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RequestEntry);
  RequestEntry(int64_t job_id, const RequestDesc& desc);
  ~RequestEntry() = default;

  const RequestDesc& desc() const { return desc_; }
  int32_t LocalRankCount() const { return local_rank2global_rank_.size(); }
  int64_t job_id() const { return job_id_; }
  int32_t LocalRankToGlobalRank(int32_t local_rank) const {
    return local_rank2global_rank_.at(local_rank);
  }
  int32_t GlobalRankToLocalRank(int32_t global_rank) const {
    return global_rank2local_rank_.at(global_rank);
  }
  bool HasRankOnThisNode() const { return !local_rank2global_rank_.empty(); }
  int32_t NodeCount() const { return node_count_; }

  bool AddRuntimeRequest(int32_t local_rank,
                         std::shared_ptr<const RuntimeRequestInfo> runtime_request_info);
  const std::shared_ptr<const RuntimeRequestInfo>& GetRuntimeRequest(int32_t local_rank);
  void ResetRuntimeRequest();

 private:
  std::mutex mutex_;
  int64_t job_id_;
  RequestDesc desc_;
  int32_t node_count_;
  std::vector<std::shared_ptr<const RuntimeRequestInfo>> runtime_request_info_vec_;
  int32_t runtime_request_info_count_;
  std::vector<DeviceDesc> local_device_vec_;
  std::vector<int64_t> local_rank2global_rank_;
  std::map<int64_t, int64_t> global_rank2local_rank_;
};

class RequestStore {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RequestStore);
  explicit RequestStore(const CollectiveBoxingPlan& collective_boxing_plan);
  ~RequestStore() = default;

  RequestEntry* MutRequestEntry(int32_t request_id) {
    return request_entry_vec_.at(request_id).get();
  }
  int32_t RequestCount() const;
  int32_t MaxMultiNodeRequestId() const;
  int32_t GetRequestIdByName(const std::string& name) const;

 private:
  std::vector<std::unique_ptr<RequestEntry>> request_entry_vec_;
  int32_t max_multi_node_request_id_ = 0;
  HashMap<std::string, int32_t> name2request_id_;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_REQUEST_STORE_H_
