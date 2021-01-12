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
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_STATIC_GROUP_COORDINATOR_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_STATIC_GROUP_COORDINATOR_H_

#include "oneflow/core/job/collective_boxing/coordinator.h"

namespace oneflow {

class CollectiveBoxingPlan;

namespace boxing {

namespace collective {

class RequestStore;
class Executor;

class StaticGroupCoordinator : public Coordinator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StaticGroupCoordinator);
  StaticGroupCoordinator() = default;
  ~StaticGroupCoordinator() override = default;

  void Init(const CollectiveBoxingPlan& collective_boxing_plan,
            std::shared_ptr<RequestStore> request_store,
            std::shared_ptr<Executor> executor) override;
  void AddRequest(int32_t request_id) override;

 private:
  struct GroupState {
    explicit GroupState(std::set<int32_t> request_ids)
        : request_ids(std::move(request_ids)), ready_request_ids() {}
    const std::set<int32_t> request_ids;
    std::set<int32_t> ready_request_ids;

    void AddReadyRequest(int32_t request_id);
    bool IsReady() const;
  };

  void DumpSummary() const;

  std::shared_ptr<RequestStore> request_store_;
  std::shared_ptr<Executor> executor_;
  std::mutex mutex_;
  std::map<int64_t, std::vector<int64_t>> job_id2group_ids_;
  std::vector<GroupState> group_id2group_state_;
  std::vector<int64_t> request_id2group_id_;
  int64_t current_job_id_ = -1;
  int64_t current_group_idx_in_job_ = -1;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_STATIC_GROUP_COORDINATOR_H_
