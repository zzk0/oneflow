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
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_NCCL_EXECUTOR_BACKEND_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_NCCL_EXECUTOR_BACKEND_H_

#include "oneflow/core/job/collective_boxing/executor.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"

namespace oneflow {

namespace boxing {

namespace collective {

class NcclExecutorBackend : public ExecutorBackend {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclExecutorBackend)
  NcclExecutorBackend();
  ~NcclExecutorBackend() override;

 private:
  void Init(const CollectiveBoxingPlan& collective_boxing_plan,
            std::shared_ptr<RequestStore> request_store) override;
  void GroupRequests(const std::vector<int32_t>& request_ids,
                     std::vector<std::vector<int32_t>>* groups) override;
  void ExecuteRequests(const std::vector<int32_t>& request_ids) override;

  struct Event {
    int64_t device_id;
    cudaEvent_t cuda_event;
    std::function<void(Maybe<void>)> callback;
  };

  struct NcclDeviceCtx : public DeviceCtx {
    const cudaStream_t& cuda_stream() const override { return stream; }
    void AddCallBack(std::function<void()>) const override { UNIMPLEMENTED(); }

    cudaStream_t stream = nullptr;
    char* fusion_buffer = nullptr;
  };

  int32_t num_devices_;
  int64_t num_streams_;
  int64_t fusion_threshold_;
  const CollectiveBoxingConf collective_boxing_conf_;

  HashMap<DeviceSet, std::vector<std::map<int64_t, ncclComm_t>>>
      device_set2stream_id2device_id2comm_;
  std::vector<std::map<int64_t, std::unique_ptr<NcclDeviceCtx>>> stream_id2device_id2device_ctx_;
  std::list<Event> event_list_;
  std::thread event_list_poll_thread_;
  std::mutex event_list_mutex_;
  std::condition_variable event_list_cond_;
  bool shutdown_;
  std::mutex mutex_;
  std::shared_ptr<ThreadPool> callback_executor_pool_;

  int64_t current_stream_id_ = 0;
  std::shared_ptr<RequestStore> request_store_;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_NCCL_EXECUTOR_BACKEND_H_
