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
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_EXECUTOR_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_EXECUTOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {

namespace boxing {

namespace collective {

struct RuntimeRequestInfo {
  const void* send_buff;
  void* recv_buff;
  std::shared_ptr<const std::function<void(const Maybe<void>&)>> callback;
};

class RequestStore {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RequestStore);
  explicit RequestStore(const CollectiveBoxingPlan& collective_boxing_plan);
  ~RequestStore() = default;

  int32_t RequestCount() const;
  int32_t MaxMultiNodeRequestId() const;
  const RequestDesc& GetRequestDesc(int32_t request_id) const;
  int32_t GetLocalRankCount(int32_t request_id) const;
  int32_t GetRequestIdByName(const std::string& name) const;
  int64_t GetJobId(int32_t request_id) const;
  int64_t GetGlobalRank(int32_t request_id, int32_t local_rank) const;
  int64_t GetLocalRank(int32_t request_id, int32_t global_rank) const;
  bool HasRankOnThisNode(int32_t request_id) const;

  bool SetRuntimeRequest(int32_t request_id, int32_t local_rank,
                         std::shared_ptr<const RuntimeRequestInfo> runtime_request_info);
  const std::shared_ptr<const RuntimeRequestInfo>& GetRuntimeRequest(int32_t request_id,
                                                                     int32_t local_rank) const;
  void ResetRuntimeRequest(int32_t request_id);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class CollectiveBoxingExecutorBackend {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingExecutorBackend);
  CollectiveBoxingExecutorBackend() = default;
  virtual ~CollectiveBoxingExecutorBackend() = default;

  virtual void Init(const CollectiveBoxingPlan& collective_boxing_plan,
                    std::shared_ptr<RequestStore> request_store){};
  virtual void GroupRequests(const std::vector<int32_t>& request_ids,
                             std::vector<std::vector<int32_t>>* groups);
  virtual void ExecuteGroup(const std::vector<int32_t>& request_ids) = 0;
};

class Scheduler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Scheduler);
  ~Scheduler() = default;

  void Schedule(const RankDesc& rank_desc, std::shared_ptr<const RuntimeRequestInfo> request_info);

 private:
  friend class Global<Scheduler>;
  explicit Scheduler(const Plan& plan);

  void Init();
  void DumpSummary() const;

  struct GroupState {
    GroupState(CollectiveBoxingExecutorBackend* backend, std::set<int32_t> request_ids)
        : backend(backend), request_ids(std::move(request_ids)), ready_request_ids() {}
    CollectiveBoxingExecutorBackend* const backend;
    const std::set<int32_t> request_ids;
    std::set<int32_t> ready_request_ids;

    void AddReadyRequest(int32_t request_id);
    bool IsReady() const;
  };

  std::mutex mutex_;

  const CollectiveBoxingPlan collective_boxing_plan_;
  std::map<Backend, std::unique_ptr<CollectiveBoxingExecutorBackend>> backends_;
  std::map<int64_t, std::vector<int64_t>> job_id2group_ids_;
  std::vector<GroupState> group_id2group_state_;
  std::vector<int64_t> request_id2group_id_;
  int64_t current_job_id_ = -1;
  int64_t current_group_idx_in_job_ = -1;

  std::shared_ptr<RequestStore> request_store_;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_EXECUTOR_H_
