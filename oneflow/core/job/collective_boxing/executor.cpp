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
#include "oneflow/core/job/collective_boxing/executor.h"
#include "oneflow/core/job/collective_boxing/request_store.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/collective_boxing/nccl_executor_backend.h"

namespace oneflow {

namespace boxing {

namespace collective {

namespace {

void SortRequestIdsByOrder(RequestStore* request_store, std::vector<int32_t>* requests) {
  std::sort(requests->begin(), requests->end(), [request_store](int32_t a, int32_t b) {
    return request_store->MutRequestEntry(a)->desc().order()
           < request_store->MutRequestEntry(b)->desc().order();
  });
}

}  // namespace

void ExecutorBackend::GroupRequests(const std::vector<int32_t>& request_ids,
                                    std::vector<std::vector<int32_t>>* groups) {
  for (const int32_t request_id : request_ids) {
    groups->emplace_back(std::vector<int32_t>({request_id}));
  }
}

class RequestHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RequestHandle)
  RequestHandle(int32_t request_id, int64_t local_rank)
      : request_id_(request_id), local_rank_(local_rank) {}
  ~RequestHandle() = default;

  int32_t request_id() const { return request_id_; }

  int64_t local_rank() const { return local_rank_; }

 private:
  int32_t request_id_;
  int64_t local_rank_;
};

class ExecutorImpl : public Executor {
 public:
  ExecutorImpl() = default;
  ~ExecutorImpl() override = default;

  void Init(const CollectiveBoxingPlan& collective_boxing_plan,
            std::shared_ptr<RequestStore> request_store) override;
  void GroupRequests(const std::vector<int32_t>& request_ids,
                     std::vector<std::vector<int32_t>>* groups) override;
  void ExecuteRequests(const std::vector<int32_t>& request_ids) override;

 private:
  Backend GetUniqueBackend(const std::vector<int32_t>& request_ids);

  std::map<Backend, std::unique_ptr<ExecutorBackend>> backends_;
  std::shared_ptr<RequestStore> request_store_;
};

void ExecutorImpl::Init(const CollectiveBoxingPlan& collective_boxing_plan,
                        std::shared_ptr<RequestStore> request_store) {
  request_store_ = request_store;
#ifdef WITH_CUDA
  auto it = backends_.emplace(Backend::kBackendNCCL, std::make_unique<NcclExecutorBackend>()).first;
  it->second->Init(collective_boxing_plan, request_store_);
#endif
}

void ExecutorImpl::GroupRequests(const std::vector<int32_t>& request_ids,
                                 std::vector<std::vector<int32_t>>* groups) {
  if (request_ids.empty()) { return; }
  const Backend backend = GetUniqueBackend(request_ids);
  backends_.at(backend)->GroupRequests(request_ids, groups);
}

void ExecutorImpl::ExecuteRequests(const std::vector<int32_t>& request_ids) {
  if (request_ids.empty()) { return; }
  const Backend backend = GetUniqueBackend(request_ids);
  backends_.at(backend)->ExecuteRequests(request_ids);
}

Backend ExecutorImpl::GetUniqueBackend(const std::vector<int32_t>& request_ids) {
  const Backend backend =
      request_store_->MutRequestEntry(request_ids.front())->desc().op_desc().backend();
  for (int64_t i = 1; i < request_ids.size(); ++i) {
    CHECK_EQ(request_store_->MutRequestEntry(request_ids.at(i))->desc().op_desc().backend(),
             backend);
  }
  return backend;
}

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

void StaticGroupCoordinator::Init(const CollectiveBoxingPlan& collective_boxing_plan,
                                  std::shared_ptr<RequestStore> request_store,
                                  std::shared_ptr<Executor> executor) {
  request_store_ = request_store;
  executor_ = executor;
  const CollectiveBoxingConf collective_boxing_conf =
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  HashMap<int64_t, std::vector<int32_t>> job_id2request_ids;
  const int32_t request_count = request_store_->RequestCount();
  for (int32_t request_id = 0; request_id < request_count; ++request_id) {
    auto* request_entry = request_store_->MutRequestEntry(request_id);
    if (request_entry->HasRankOnThisNode()) {
      job_id2request_ids[request_entry->job_id()].push_back(request_id);
    }
  }
  const auto& GetRequestDesc = [&](int32_t request_id) -> const RequestDesc& {
    return request_store_->MutRequestEntry(request_id)->desc();
  };
  request_id2group_id_.resize(request_store_->RequestCount());
  for (auto& job_id7request_ids : job_id2request_ids) {
    const int64_t job_id = job_id7request_ids.first;
    auto& request_ids = job_id7request_ids.second;
    SortRequestIdsByOrder(request_store_.get(), &request_ids);
    CHECK(std::adjacent_find(request_ids.begin(), request_ids.end(),
                             [&](int32_t a, int32_t b) {
                               return GetRequestDesc(a).dependency_depth()
                                      > GetRequestDesc(b).dependency_depth();
                             })
          == request_ids.end());
    std::vector<std::vector<int32_t>> rough_request_id_groups;
    for (const int32_t request_id : request_ids) {
      bool new_group = (!collective_boxing_conf.enable_fusion()) || rough_request_id_groups.empty();
      if (!new_group) {
        const auto& cur_desc = GetRequestDesc(request_id);
        const auto& group_desc = GetRequestDesc(rough_request_id_groups.back().front());
        if (cur_desc.dependency_depth() != group_desc.dependency_depth()
            || cur_desc.op_desc().backend() != group_desc.op_desc().backend()
            || cur_desc.device_set() != group_desc.device_set()) {
          new_group = true;
        }
      }
      if (new_group) {
        rough_request_id_groups.emplace_back(std::vector<int32_t>({request_id}));
      } else {
        rough_request_id_groups.back().push_back(request_id);
      }
    }
    for (const auto& rough_group : rough_request_id_groups) {
      std::vector<std::vector<int32_t>> groups;
      executor_->GroupRequests(rough_group, &groups);
      for (const auto& group : groups) {
        const int64_t group_id = group_id2group_state_.size();
        group_id2group_state_.emplace_back(std::set<int32_t>({group.cbegin(), group.cend()}));
        job_id2group_ids_[job_id].push_back(group_id);
        for (int32_t r : group) { request_id2group_id_[r] = group_id; }
      }
    }
  }
  DumpSummary();
}

void StaticGroupCoordinator::AddRequest(int32_t request_id) {
  const int64_t job_id = request_store_->MutRequestEntry(request_id)->job_id();
  std::unique_lock<std::mutex> lock(mutex_);
  if (current_job_id_ == -1) {
    current_job_id_ = job_id;
    current_group_idx_in_job_ = 0;
  } else {
    CHECK_EQ(current_job_id_, job_id);
  }
  group_id2group_state_.at(request_id2group_id_.at(request_id)).AddReadyRequest(request_id);
  const std::vector<int64_t>& group_ids = job_id2group_ids_.at(current_job_id_);
  int64_t num_launched_groups = 0;
  while (true) {
    const int64_t group_id = group_ids.at(current_group_idx_in_job_);
    auto& group_state = group_id2group_state_.at(group_id);
    if (group_state.IsReady()) {
      executor_->ExecuteRequests(
          std::vector<int32_t>({group_state.request_ids.cbegin(), group_state.request_ids.cend()}));
      group_state.ready_request_ids.clear();
      current_group_idx_in_job_ = (current_group_idx_in_job_ + 1) % group_ids.size();
      num_launched_groups += 1;
    } else {
      break;
    }
  }
  if (current_group_idx_in_job_ == 0 && num_launched_groups > 0) {
    current_job_id_ = -1;
    current_group_idx_in_job_ = -1;
  }
}

void StaticGroupCoordinator::GroupState::AddReadyRequest(int32_t request_id) {
  CHECK(request_ids.find(request_id) != request_ids.end());
  CHECK(ready_request_ids.emplace(request_id).second);
}

void StaticGroupCoordinator::DumpSummary() const {
  if (!Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) { return; }
  auto group_ls = TeePersistentLogStream::Create("boxing/collective/group");
  for (int64_t group_id = 0; group_id < group_id2group_state_.size(); ++group_id) {
    group_ls << "group id: " << std::to_string(group_id) << "\n";
    for (const int32_t request_id : group_id2group_state_.at(group_id).request_ids) {
      group_ls->Write(request_store_->MutRequestEntry(request_id)->desc());
    }
  }
}

bool StaticGroupCoordinator::GroupState::IsReady() const {
  return ready_request_ids.size() == request_ids.size();
}

struct Scheduler::Impl {
  explicit Impl(const CollectiveBoxingPlan& collective_boxing_plan);

  CollectiveBoxingPlan collective_boxing_plan;
  std::shared_ptr<RequestStore> request_store;
  std::shared_ptr<Coordinator> coordinator;
};

Scheduler::Impl::Impl(const CollectiveBoxingPlan& collective_boxing_plan)
    : collective_boxing_plan(collective_boxing_plan) {
  request_store.reset(new RequestStore(collective_boxing_plan));
  std::shared_ptr<Executor> executor(new ExecutorImpl());
  executor->Init(collective_boxing_plan, request_store);
  coordinator.reset(new StaticGroupCoordinator());
  coordinator->Init(collective_boxing_plan, request_store, executor);
}

Scheduler::Scheduler(const Plan& plan) { impl_.reset(new Impl(plan.collective_boxing_plan())); }

std::shared_ptr<RequestHandle> Scheduler::CreateRequestHandle(const RankDesc& rank_desc) {
  const int32_t request_id = impl_->request_store->GetRequestIdByName(rank_desc.op_desc().name());
  auto* request_entry = impl_->request_store->MutRequestEntry(request_id);
  CHECK(rank_desc.op_desc() == request_entry->desc().op_desc());
  const int64_t local_rank = request_entry->GlobalRankToLocalRank(rank_desc.rank());
  return std::make_shared<RequestHandle>(request_id, local_rank);
}

void Scheduler::Schedule(const std::shared_ptr<RequestHandle>& handle,
                         std::shared_ptr<const RuntimeRequestInfo> request_info) {
  const int64_t request_id = handle->request_id();
  const int64_t local_rank = handle->local_rank();
  const bool ready = impl_->request_store->MutRequestEntry(request_id)
                         ->AddRuntimeRequest(local_rank, std::move(request_info));
  if (ready) { impl_->coordinator->AddRequest(request_id); }
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
