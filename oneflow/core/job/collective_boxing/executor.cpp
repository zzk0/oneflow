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
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/kernel/batch_memcpy_kernel_util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/thread/thread_pool.h"
#ifdef WITH_CUDA
#include <nccl.h>

#include <utility>
#endif

namespace oneflow {

namespace boxing {

namespace collective {

namespace {

#ifdef WITH_CUDA
ncclRedOp_t GetNcclReduceOp(ReduceMethod reduce_method) {
  if (reduce_method == kReduceMethodSum) {
    return ncclRedOp_t::ncclSum;
  } else {
    UNIMPLEMENTED();
  }
}
#endif

void SortRequestsByOrder(std::vector<const RequestDesc*>* requests) {
  std::sort(requests->begin(), requests->end(),
            [](const RequestDesc* a, const RequestDesc* b) { return a->order() < b->order(); });
}

void SortRequestIdsByOrder(RequestStore* request_store, std::vector<int32_t>* requests) {
  std::sort(requests->begin(), requests->end(), [request_store](int32_t a, int32_t b) {
    return request_store->MutRequestEntry(a)->desc().order()
           < request_store->MutRequestEntry(b)->desc().order();
  });
}

bool IsDeviceOnThisMachine(const DeviceDesc& device_desc) {
  return device_desc.machine_id() == Global<MachineCtx>::Get()->this_machine_id();
}

bool HasDeviceOnThisMachine(const DeviceSet& device_set) {
  return std::any_of(
      device_set.device().cbegin(), device_set.device().cend(),
      [](const DeviceDesc& device_desc) { return IsDeviceOnThisMachine(device_desc); });
}

std::string GetNcclUniqueIdRpcKey(const std::string& name, int64_t stream_id) {
  return "CollectiveBoxingExecutorNcclUniqueIdRpcKey-" + name + "-" + std::to_string(stream_id);
}

int64_t GetRequestSize(const RequestDesc& request) {
  return Shape(request.op_desc().shape()).elem_cnt()
         * GetSizeOfDataType(request.op_desc().data_type());
}

int64_t GetAlignedRequestSize(const RequestDesc& request) {
  return GetCudaAlignedSize(GetRequestSize(request));
}

}  // namespace

#ifdef WITH_CUDA

void ExecutorBackend::GroupRequests(const std::vector<int32_t>& request_ids,
                                    std::vector<std::vector<int32_t>>* groups) {
  for (const int32_t request_id : request_ids) {
    groups->emplace_back(std::vector<int32_t>({request_id}));
  }
}

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

NcclExecutorBackend::NcclExecutorBackend()
    : collective_boxing_conf_(Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf()),
      shutdown_(false) {
  OF_CUDA_CHECK(cudaGetDeviceCount(&num_devices_));
  callback_executor_pool_.reset(new ThreadPool(num_devices_));
  CHECK_GT(collective_boxing_conf_.nccl_num_streams(), 0);
  num_streams_ = collective_boxing_conf_.nccl_num_streams();
  CHECK_GE(collective_boxing_conf_.nccl_fusion_threshold_mb(), 0);
  fusion_threshold_ = collective_boxing_conf_.nccl_fusion_threshold_mb() * 1024 * 1024;
  event_list_poll_thread_ = std::thread([this]() {
    std::list<Event> local_event_list;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(event_list_mutex_);
        if (local_event_list.empty()) {
          event_list_cond_.wait(lock, [this]() { return (!event_list_.empty()) || shutdown_; });
        }
        local_event_list.splice(local_event_list.end(), event_list_);
      }
      if (local_event_list.empty() && shutdown_) { break; }
      for (auto it = local_event_list.begin(); it != local_event_list.end();) {
        OF_CUDA_CHECK(cudaSetDevice(it->device_id));
        cudaError_t err = cudaEventQuery(it->cuda_event);
        if (err == cudaErrorNotReady) {
          ++it;
          continue;
        } else if (err == cudaSuccess) {
          OF_CUDA_CHECK(cudaEventDestroy(it->cuda_event));
          auto callback_ptr =
              std::make_shared<std::function<void(Maybe<void>)>>(std::move(it->callback));
          callback_executor_pool_->AddWork(
              [callback_ptr]() { (*callback_ptr)(Maybe<void>::Ok()); });
          local_event_list.erase(it++);
        } else {
          OF_CUDA_CHECK(err);
          UNIMPLEMENTED();
        }
      }
    }
  });
}

NcclExecutorBackend::~NcclExecutorBackend() {
  {
    std::unique_lock<std::mutex> lock(event_list_mutex_);
    shutdown_ = true;
    event_list_cond_.notify_all();
  }
  event_list_poll_thread_.join();
  callback_executor_pool_.reset();
  CudaCurrentDeviceGuard guard;
  for (auto& device_id2device_ctx : stream_id2device_id2device_ctx_) {
    for (auto& device_id7device_ctx : device_id2device_ctx) {
      OF_CUDA_CHECK(cudaSetDevice(device_id7device_ctx.first));
      OF_CUDA_CHECK(cudaStreamSynchronize(device_id7device_ctx.second->stream));
      OF_CUDA_CHECK(cudaStreamDestroy(device_id7device_ctx.second->stream));
      OF_CUDA_CHECK(cudaFree(device_id7device_ctx.second->fusion_buffer));
    }
  }
  for (auto& device_set7stream_id2device_id2comm : device_set2stream_id2device_id2comm_) {
    for (auto& device_id2comm : device_set7stream_id2device_id2comm.second) {
      for (auto& device_id7comm : device_id2comm) {
        OF_CUDA_CHECK(cudaSetDevice(device_id7comm.first));
        OF_NCCL_CHECK(ncclCommDestroy(device_id7comm.second));
      }
    }
  }
}

void NcclExecutorBackend::GroupRequests(const std::vector<int32_t>& request_ids,
                                        std::vector<std::vector<int32_t>>* groups) {
  std::vector<int32_t> group;
  int64_t group_size = 0;
  auto IsOpFusionEnabled = [&](const RequestDesc& request) -> bool {
    const OpType op_type = request.op_desc().op_type();
    if (op_type == OpType::kOpTypeAllReduce) {
      return collective_boxing_conf_.nccl_fusion_all_reduce();
    } else if (op_type == OpType::kOpTypeAllGather) {
      return collective_boxing_conf_.nccl_fusion_all_gather();
    } else if (op_type == OpType::kOpTypeReduceScatter) {
      return collective_boxing_conf_.nccl_fusion_reduce_scatter();
    } else if (op_type == OpType::kOpTypeReduce) {
      return collective_boxing_conf_.nccl_fusion_reduce();
    } else if (op_type == OpType::kOpTypeBroadcast) {
      return collective_boxing_conf_.nccl_fusion_broadcast();
    } else if (op_type == OpType::kOpTypeAll2All) {
      return false;
    } else {
      UNIMPLEMENTED();
      return false;
    }
  };
  auto CanFuse = [&](const RequestDesc& lhs, const RequestDesc& rhs) -> bool {
    const bool enable_mixed_fusion = (!collective_boxing_conf_.nccl_fusion_all_reduce_use_buffer())
                                     && collective_boxing_conf_.nccl_enable_mixed_fusion();
    if (lhs.device_set() != rhs.device_set()) { return false; }
    if (!IsOpFusionEnabled(lhs) || !IsOpFusionEnabled(rhs)) { return false; }
    if (lhs.op_desc().op_type() != rhs.op_desc().op_type() && (!enable_mixed_fusion)) {
      return false;
    }
    const OpType op_type = lhs.op_desc().op_type();
    if (op_type == OpType::kOpTypeAllReduce) {
      if (collective_boxing_conf_.nccl_fusion_all_reduce_use_buffer()) {
        CHECK(lhs.op_desc().has_reduce_method());
        CHECK(rhs.op_desc().has_reduce_method());
        return lhs.op_desc().reduce_method() == rhs.op_desc().reduce_method()
               && lhs.op_desc().data_type() == rhs.op_desc().data_type();
      } else {
        return true;
      }
    } else if (op_type == OpType::kOpTypeReduce || op_type == OpType::kOpTypeBroadcast
               || op_type == OpType::kOpTypeReduceScatter || op_type == OpType::kOpTypeAllGather) {
      return true;
    } else if (op_type == OpType::kOpTypeAll2All) {
      return false;
    } else {
      UNIMPLEMENTED();
      return false;
    }
  };

  for (const int32_t request_id : request_ids) {
    const auto& request = request_store_->MutRequestEntry(request_id)->desc();
    const int64_t size = GetAlignedRequestSize(request);
    if (group.empty() || !CanFuse(request_store_->MutRequestEntry(group.back())->desc(), request)
        || group_size + size > fusion_threshold_
        || group.size() >= collective_boxing_conf_.nccl_fusion_max_ops()) {
      if (!group.empty()) {
        groups->emplace_back();
        groups->back().swap(group);
        group_size = 0;
      }
    }
    group.push_back(request_id);
    group_size += size;
  }
  if (!group.empty()) {
    groups->emplace_back();
    groups->back().swap(group);
  }
}

void NcclExecutorBackend::ExecuteRequests(const std::vector<int32_t>& request_ids) {
  std::vector<const RequestDesc*> group;
  std::vector<std::map<int64_t, std::shared_ptr<const RuntimeRequestInfo>>> ranks;
  group.reserve(request_ids.size());
  for (const int32_t request_id : request_ids) {
    auto* request_entry = request_store_->MutRequestEntry(request_id);
    group.push_back(&request_entry->desc());
    ranks.emplace_back();
    for (int32_t local_rank = 0; local_rank < request_entry->LocalRankCount(); ++local_rank) {
      ranks.back()[request_entry->LocalRankToGlobalRank(local_rank)] =
          request_entry->GetRuntimeRequest(local_rank);
    }
    request_entry->ResetRuntimeRequest();
  }
  CHECK_EQ(group.size(), ranks.size());
  if (group.empty()) { return; }
  const int64_t group_size = group.size();
  std::map<int64_t, std::vector<std::shared_ptr<const std::function<void(const Maybe<void>&)>>>>
      device_id2callbacks;
  const int64_t stream_id = current_stream_id_;
  current_stream_id_ = (current_stream_id_ + 1) % num_streams_;
  CudaCurrentDeviceGuard device_guard;
  auto& device_id2comm =
      device_set2stream_id2device_id2comm_.size() == 1
          ? device_set2stream_id2device_id2comm_.begin()->second.at(stream_id)
          : device_set2stream_id2device_id2comm_.at(group.front()->device_set()).at(stream_id);
  auto& device_id2device_ctx = stream_id2device_id2device_ctx_.at(stream_id);
  if (group.front()->op_desc().op_type() == OpType::kOpTypeAllReduce
      && collective_boxing_conf_.nccl_fusion_all_reduce_use_buffer() && group.size() > 1) {
    int64_t offset = 0;
    std::map<int64_t, std::vector<MemcpyParam>> device_id2copy_in_params;
    std::map<int64_t, std::vector<MemcpyParam>> device_id2copy_out_params;
    for (int64_t i = 0; i < group.size(); ++i) {
      const RequestDesc* request_desc = group.at(i);
      if (i != 0) {
        CHECK_EQ(request_desc->op_desc().reduce_method(), group.front()->op_desc().reduce_method());
        CHECK_EQ(request_desc->op_desc().data_type(), group.front()->op_desc().data_type());
      }
      const std::map<int64_t, std::shared_ptr<const RuntimeRequestInfo>>& rank2request_info =
          ranks.at(i);
      const int64_t size = GetRequestSize(*request_desc);
      CHECK_LE(offset + size, fusion_threshold_);
      const int64_t aligned_size = GetCudaAlignedSize(size);
      for (const auto& rank7request_info : rank2request_info) {
        const int64_t rank = rank7request_info.first;
        const RuntimeRequestInfo& request_info = *rank7request_info.second;
        const DeviceDesc& device_desc = request_desc->device_set().device().Get(rank);
        const int64_t device_id = device_desc.device_id();
        auto& device_ctx = device_id2device_ctx.at(device_id);
        device_id2copy_in_params[device_id].push_back(MemcpyParam{
            .dst = device_ctx->fusion_buffer + offset,
            .src = request_info.send_buff,
            .count = static_cast<size_t>(size),
        });
        device_id2copy_out_params[device_id].push_back(MemcpyParam{
            .dst = request_info.recv_buff,
            .src = device_ctx->fusion_buffer + offset,
            .count = static_cast<size_t>(size),
        });
        device_id2callbacks[device_id].reserve(group_size);
        device_id2callbacks[device_id].push_back(request_info.callback);
      }
      offset += aligned_size;
    }
    for (auto& device_id7copy_in_params : device_id2copy_in_params) {
      OF_CUDA_CHECK(cudaSetDevice(device_id7copy_in_params.first));
      BatchMemcpyKernelUtil<DeviceType::kGPU>::Copy(
          device_id2device_ctx.at(device_id7copy_in_params.first).get(),
          device_id7copy_in_params.second);
    }
    OF_NCCL_CHECK(ncclGroupStart());
    const int64_t size_of_data_type = GetSizeOfDataType(group.front()->op_desc().data_type());
    CHECK_EQ(offset % size_of_data_type, 0);
    const int64_t elem_cnt = offset / size_of_data_type;
    for (auto& device_id7comm : device_id2comm) {
      OF_CUDA_CHECK(cudaSetDevice(device_id7comm.first));
      auto& device_ctx = device_id2device_ctx.at(device_id7comm.first);
      OF_NCCL_CHECK(ncclAllReduce(device_ctx->fusion_buffer, device_ctx->fusion_buffer, elem_cnt,
                                  GetNcclDataType(group.front()->op_desc().data_type()),
                                  GetNcclReduceOp(group.front()->op_desc().reduce_method()),
                                  device_id7comm.second, device_ctx->stream));
    }
    OF_NCCL_CHECK(ncclGroupEnd());
    for (auto& device_id7copy_out_params : device_id2copy_out_params) {
      OF_CUDA_CHECK(cudaSetDevice(device_id7copy_out_params.first));
      BatchMemcpyKernelUtil<DeviceType::kGPU>::Copy(
          device_id2device_ctx.at(device_id7copy_out_params.first).get(),
          device_id7copy_out_params.second);
    }
  } else {
    OF_NCCL_CHECK(ncclGroupStart());
    for (int64_t i = 0; i < group.size(); ++i) {
      const RequestDesc* request_desc = group.at(i);
      const OpDesc& op_desc = request_desc->op_desc();
      const std::map<int64_t, std::shared_ptr<const RuntimeRequestInfo>>& rank2request_info =
          ranks.at(i);
      for (const auto& rank7request_info : rank2request_info) {
        const int64_t rank = rank7request_info.first;
        const RuntimeRequestInfo& request_info = *rank7request_info.second;
        const DeviceDesc& device_desc = request_desc->device_set().device().Get(rank);
        const int64_t device_id = device_desc.device_id();
        OF_CUDA_CHECK(cudaSetDevice(device_id));
        ncclComm_t comm = device_id2comm.at(device_id);
        auto& device_ctx = device_id2device_ctx.at(device_id);
        ncclDataType_t nccl_data_type = GetNcclDataType(op_desc.data_type());
        const OpType op_type = op_desc.op_type();
        const int64_t num_ranks = op_desc.num_ranks();
        const int64_t elem_cnt = Shape(op_desc.shape()).elem_cnt();
        const void* send_buff = request_info.send_buff;
        void* recv_buff = request_info.recv_buff;
        device_id2callbacks[device_id].reserve(group_size);
        device_id2callbacks[device_id].push_back(request_info.callback);
        if (op_type == OpType::kOpTypeAllReduce) {
          OF_NCCL_CHECK(ncclAllReduce(send_buff, recv_buff, elem_cnt, nccl_data_type,
                                      GetNcclReduceOp(op_desc.reduce_method()), comm,
                                      device_ctx->stream));
        } else if (op_type == OpType::kOpTypeAllGather) {
          CHECK_EQ(elem_cnt % num_ranks, 0);
          OF_NCCL_CHECK(ncclAllGather(send_buff, recv_buff, elem_cnt / num_ranks, nccl_data_type,
                                      comm, device_ctx->stream));
        } else if (op_type == OpType::kOpTypeReduceScatter) {
          CHECK_EQ(elem_cnt % num_ranks, 0);
          OF_NCCL_CHECK(ncclReduceScatter(send_buff, recv_buff, elem_cnt / num_ranks,
                                          nccl_data_type, GetNcclReduceOp(op_desc.reduce_method()),
                                          comm, device_ctx->stream));
        } else if (op_type == OpType::kOpTypeReduce) {
          OF_NCCL_CHECK(ncclReduce(send_buff, recv_buff, elem_cnt, nccl_data_type,
                                   GetNcclReduceOp(op_desc.reduce_method()), op_desc.root(), comm,
                                   device_ctx->stream));
        } else if (op_type == OpType::kOpTypeBroadcast) {
          OF_NCCL_CHECK(ncclBroadcast(send_buff, recv_buff, elem_cnt, nccl_data_type,
                                      op_desc.root(), comm, device_ctx->stream));
        } else if (op_type == OpType::kOpTypeAll2All) {
#if NCCL_VERSION_CODE > 2700
          const int64_t elem_per_rank = elem_cnt / num_ranks;
          const int64_t elem_per_chunk = elem_per_rank / num_ranks;
          const int64_t dtype_size = GetSizeOfDataType(op_desc.data_type());
          const int64_t chunk_size = elem_per_chunk * dtype_size;
          for (int64_t j = 0; j < num_ranks; ++j) {
            OF_NCCL_CHECK(ncclSend(reinterpret_cast<const void*>(
                                       reinterpret_cast<const char*>(send_buff) + j * chunk_size),
                                   elem_per_chunk, nccl_data_type, j, comm, device_ctx->stream));
            OF_NCCL_CHECK(ncclRecv(
                reinterpret_cast<void*>(reinterpret_cast<char*>(recv_buff) + j * chunk_size),
                elem_per_chunk, nccl_data_type, j, comm, device_ctx->stream));
          }
#else
          UNIMPLEMENTED();
#endif
        } else {
          UNIMPLEMENTED();
        }
      }
    }
    OF_NCCL_CHECK(ncclGroupEnd());
  }
  for (auto& device_id7callbacks : device_id2callbacks) {
    const int64_t device_id = device_id7callbacks.first;
    OF_CUDA_CHECK(cudaSetDevice(device_id));
    cudaEvent_t event;
    OF_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    OF_CUDA_CHECK(cudaEventRecord(event, device_id2device_ctx.at(device_id)->stream));
    {
      std::unique_lock<std::mutex> event_list_lock(event_list_mutex_);
      event_list_.emplace_back(Event{device_id, event, [=](const Maybe<void>& status) {
                                       for (const auto& callback : device_id7callbacks.second) {
                                         (*callback)(status);
                                       }
                                     }});
      event_list_cond_.notify_all();
    }
  }
}

void NcclExecutorBackend::Init(const CollectiveBoxingPlan& collective_boxing_plan,
                               std::shared_ptr<RequestStore> request_store) {
  request_store_ = request_store;
  CudaCurrentDeviceGuard guard;
  std::set<int64_t> local_device_ids;
  for (const auto& job_id7request_set : collective_boxing_plan.job_id2request_set()) {
    std::vector<const RequestDesc*> requests;
    for (const RequestDesc& request : job_id7request_set.second.request()) {
      if (request.op_desc().backend() == Backend::kBackendNCCL) { requests.push_back(&request); }
    }
    SortRequestsByOrder(&requests);
    for (const RequestDesc* request : requests) {
      std::set<int64_t> local_ranks;
      const DeviceSet& device_set = request->device_set();
      for (int64_t i = 0; i < device_set.device_size(); ++i) {
        const DeviceDesc& device_desc = device_set.device(i);
        if (IsDeviceOnThisMachine(device_desc)) {
          local_ranks.emplace(i);
          local_device_ids.emplace(device_desc.device_id());
        }
      }
      if (local_ranks.empty()) { continue; }
      if (device_set2stream_id2device_id2comm_.count(device_set) > 0) { continue; }
      auto& stream_id2device_id2comm = device_set2stream_id2device_id2comm_[device_set];
      stream_id2device_id2comm.resize(num_streams_);
      for (int64_t stream_id = 0; stream_id < num_streams_; ++stream_id) {
        auto& device_id2comm = stream_id2device_id2comm.at(stream_id);
        for (const int64_t rank : local_ranks) {
          const int64_t device_id = device_set.device(rank).device_id();
          device_id2comm.emplace(device_id, ncclComm_t{});
        }
        ncclUniqueId nccl_unique_id{};
        if (local_ranks.count(0) > 0) {
          OF_NCCL_CHECK(ncclGetUniqueId(&nccl_unique_id));
          if (local_ranks.size() != device_set.device_size()) {
            const std::string rpc_key = GetNcclUniqueIdRpcKey(request->op_desc().name(), stream_id);
            Global<CtrlClient>::Get()->PushKV(rpc_key, NcclUniqueIdToString(nccl_unique_id));
          }
        } else {
          const std::string rpc_key = GetNcclUniqueIdRpcKey(request->op_desc().name(), stream_id);
          Global<CtrlClient>::Get()->PullKV(rpc_key, [&nccl_unique_id](const std::string& val) {
            NcclUniqueIdFromString(val, &nccl_unique_id);
          });
        }
        OF_NCCL_CHECK(ncclGroupStart());
        for (const int64_t rank : local_ranks) {
          const int64_t device_id = device_set.device(rank).device_id();
          OF_CUDA_CHECK(cudaSetDevice(device_id));
          OF_NCCL_CHECK(ncclCommInitRank(&device_id2comm.at(device_id), device_set.device_size(),
                                         nccl_unique_id, rank));
        }
        OF_NCCL_CHECK(ncclGroupEnd())
            << "To see more detail, please run OneFlow with system variable NCCL_DEBUG=INFO";
      }
    }
  }
  int cuda_stream_greatest_priority;
  OF_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(nullptr, &cuda_stream_greatest_priority));
  stream_id2device_id2device_ctx_.resize(num_streams_);
  for (int64_t stream_id = 0; stream_id < num_streams_; ++stream_id) {
    auto& device_id2device_ctx_ = stream_id2device_id2device_ctx_.at(stream_id);
    for (const int64_t device_id : local_device_ids) {
      device_id2device_ctx_.emplace(device_id, std::make_unique<NcclDeviceCtx>());
    }
    for (const int64_t device_id : local_device_ids) {
      auto& device_ctx = device_id2device_ctx_.at(device_id);
      OF_CUDA_CHECK(cudaSetDevice(device_id));
      OF_CUDA_CHECK(cudaStreamCreateWithPriority(&device_ctx->stream, cudaStreamNonBlocking,
                                                 cuda_stream_greatest_priority));
      OF_CUDA_CHECK(cudaMalloc(&device_ctx->fusion_buffer, fusion_threshold_));
    }
  }
}

#endif  // WITH_CUDA

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
