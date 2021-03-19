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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

template<typename T>
void InitializeWithConf(const InitializerConf& conf, const uint32_t random_seed, Blob* blob) {
  KernelUtil<DeviceType::kCPU, T>::InitializeWithConf(nullptr, conf, random_seed, blob);
}

struct InitializeWithConfUtil final {
#define MAKE_INITIALIZE_SWITCH_ENTRY(func_name, T) func_name<T>
  DEFINE_STATIC_SWITCH_FUNC(void, InitializeWithConf, MAKE_INITIALIZE_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
#undef MAKE_INITIALIZE_SWITCH_ENTRY
};

TensorSliceView GetPartSlice(const KernelConf& kernel_conf, const int64_t parallel_id) {
  const ModelIoV2KernelConf& conf = kernel_conf.model_io_v2_conf();
  return TensorSliceView(conf.slice_view(parallel_id));
}

TensorSliceView GetPartSlice(const KernelConf& kernel_conf) {
  return GetPartSlice(kernel_conf, kernel_conf.model_io_v2_conf().parallel_ctx().parallel_id());
}

class OnDemandHostBlob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OnDemandHostBlob);
  explicit OnDemandHostBlob(const Blob* like) {
    Shape shape;
    like->shape().ToShape(&shape);
    blob_desc_.reset(new RtBlobDesc(BlobDesc(shape, like->data_type())));
    Init();
  }
  explicit OnDemandHostBlob(const RtBlobDesc& blob_desc) {
    blob_desc_.reset(new RtBlobDesc(BlobDesc(blob_desc.body_shape(), blob_desc.data_type())));
    Init();
  }
  explicit OnDemandHostBlob(const Shape& shape, DataType data_type) {
    BlobDesc blob_desc(data_type);
    blob_desc.mut_shape() = shape;
    blob_desc_.reset(new RtBlobDesc(blob_desc));
    Init();
  }
  ~OnDemandHostBlob() = default;

  Blob* blob() const { return blob_.get(); }

 private:
  void Init() {
    header.resize(blob_desc_->ByteSizeOfBlobHeader());
    data.resize(blob_desc_->AlignedByteSizeOfBlobBody());
    MemoryCase host_mem_case;
    host_mem_case.mutable_host_mem();
    blob_.reset(new Blob(host_mem_case, blob_desc_.get(), header.data(), data.data()));
  }

  std::vector<char> header;
  std::vector<char> data;
  std::unique_ptr<Blob> blob_;
  std::unique_ptr<RtBlobDesc> blob_desc_;
};

template<DeviceType device_type>
void SyncCopyToHost(DeviceCtx* ctx, const void* src, void* dst, size_t size);

template<>
void SyncCopyToHost<DeviceType::kCPU>(DeviceCtx* ctx, const void* src, void* dst, size_t size) {
  std::memcpy(dst, src, size);
}

#ifdef WITH_CUDA
template<>
void SyncCopyToHost<DeviceType::kGPU>(DeviceCtx* ctx, const void* src, void* dst, size_t size) {
  OF_CUDA_CHECK(cudaStreamSynchronize(ctx->cuda_stream()));
  OF_CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, ctx->cuda_stream()));
  OF_CUDA_CHECK(cudaStreamSynchronize(ctx->cuda_stream()));
}
#endif

template<DeviceType device_type>
void SyncCopyToDevice(DeviceCtx* ctx, const void* src, void* dst, size_t size);

template<>
void SyncCopyToDevice<DeviceType::kCPU>(DeviceCtx* ctx, const void* src, void* dst, size_t size) {
  std::memcpy(dst, src, size);
}

#ifdef WITH_CUDA
template<>
void SyncCopyToDevice<DeviceType::kGPU>(DeviceCtx* ctx, const void* src, void* dst, size_t size) {
  OF_CUDA_CHECK(cudaStreamSynchronize(ctx->cuda_stream()));
  OF_CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, ctx->cuda_stream()));
  OF_CUDA_CHECK(cudaStreamSynchronize(ctx->cuda_stream()));
}
#endif

template<DeviceType device_type>
std::string SyncReadStringFromBlob(DeviceCtx* ctx, const Blob* blob) {
  std::vector<char> content;
  const int64_t size = blob->shape().elem_cnt();
  content.resize(size);
  SyncCopyToHost<device_type>(ctx, blob->dptr(), content.data(), size);
  return std::string(content.data(), content.size());
}

std::string GetTmpPartKey(const std::string& base, const int64_t parallel_id,
                          const int64_t parallel_num) {
  return "tmp-part-" + std::to_string(parallel_id) + "-" + std::to_string(parallel_num) + "-"
         + base;
}

std::string GetTmpPartKey(const std::string& base, const ParallelContext& parallel_ctx) {
  return GetTmpPartKey(base, parallel_ctx.parallel_id(), parallel_ctx.parallel_num());
}

void HostSliceCopy(Blob* dst, const TensorSliceView& dst_slice, const Blob* src,
                   const TensorSliceView& src_slice) {
  CpuDeviceCtx cpu_device_ctx;
  std::unique_ptr<MemoryCopier> host_memory_copier(NewDefaultMemoryCopier(DeviceType::kCPU));
  TensorSliceCopier copier(dst_slice, src_slice, dst->data_type());
  copier.Copy(&cpu_device_ctx, *host_memory_copier, dst, src);
}

template<DeviceType device_type>
class AutoSyncBlobAccessor final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoSyncBlobAccessor);
  AutoSyncBlobAccessor(DeviceCtx* ctx, Blob* underlying, bool read_sync, bool write_sync)
      : device_ctx_(ctx),
        underlying_(underlying),
        read_sync_(read_sync),
        write_sync_(write_sync),
        host_blob_(underlying) {
    if (read_sync_) {
      SyncCopyToHost<device_type>(device_ctx_, underlying_->dptr(), host_blob_.blob()->mut_dptr(),
                                  underlying_->ByteSizeOfBlobBody());
    }
  }
  ~AutoSyncBlobAccessor() {
    if (write_sync_) {
      SyncCopyToDevice<device_type>(device_ctx_, host_blob_.blob()->dptr(), underlying_->mut_dptr(),
                                    underlying_->ByteSizeOfBlobBody());
    }
  }

  Blob* host_blob() { return host_blob_.blob(); }

 private:
  DeviceCtx* device_ctx_;
  Blob* underlying_;
  bool read_sync_;
  bool write_sync_;
  OnDemandHostBlob host_blob_;
};

template<>
class AutoSyncBlobAccessor<DeviceType::kCPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoSyncBlobAccessor);
  AutoSyncBlobAccessor(DeviceCtx* ctx, Blob* underlying, bool read_sync, bool write_sync)
      : underlying_(underlying) {}
  ~AutoSyncBlobAccessor() = default;

  Blob* host_blob() { return underlying_; }

 private:
  Blob* underlying_;
};

}  // namespace

template<DeviceType device_type>
class ModelInitV2Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelInitV2Kernel);
  ModelInitV2Kernel() = default;
  ~ModelInitV2Kernel() override = default;

 private:
  void VirtualKernelInit() override {
    const auto& hierarchy =
        ParallelDesc(
            this->kernel_conf().op_attribute().parallel_conf_signature().op_parallel_conf())
            .hierarchy();

    std::vector<int64_t> parallel_rank(5);
    NdIndexOffsetHelper<int64_t, 5> hierarchy_helper(hierarchy->dim_vec().data(),
                                                     hierarchy->NumAxes());
    const ParallelContext& parallel_ctx = this->kernel_conf().model_io_v2_conf().parallel_ctx();
    hierarchy_helper.OffsetToNdIndex(parallel_ctx.parallel_id(), parallel_rank.data());
    DimVector seed_vec;
    std::vector<int64_t> seed_rank;
    const auto& parallel_distribution_map = this->kernel_conf()
                                                .op_attribute()
                                                .parallel_distribution_signature()
                                                .bn_in_op2parallel_distribution();
    const auto it = parallel_distribution_map.find("ref");
    CHECK(it != parallel_distribution_map.end());
    ParallelDistribution in_parallel_distribution = it->second;
    for (int64_t i = 0; i < hierarchy->NumAxes(); ++i) {
      SbpParallel sbp_parallel = in_parallel_distribution.sbp_parallel(i);
      CHECK(sbp_parallel.has_split_parallel() || sbp_parallel.has_broadcast_parallel());
      if (sbp_parallel.has_split_parallel()) {
        seed_vec.push_back(hierarchy->At(i));
        seed_rank.push_back(parallel_rank.at(i));
      }
    }
    if (seed_vec.size() == 0) {
      seed_id_ = 0;
      seed_num_ = 1;
    } else {
      NdIndexOffsetHelper<int64_t, 5> seed_helper(seed_vec.data(), seed_vec.size());
      seed_id_ = seed_helper.NdIndexToOffset(seed_rank.data(), seed_rank.size());
      seed_num_ = Shape(seed_vec).elem_cnt();
    }
  }
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    ForwardDataContent(ctx, BnInOp2Blob);
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const ModelInitV2OpConf& conf = this->op_conf().model_init_v2_conf();
    Blob* ref = BnInOp2Blob("ref");
    const DataType data_type = ref->data_type();
    const VariableOpConf& original_variable_conf = conf.original_variable_conf();
    AutoSyncBlobAccessor<device_type> ref_accessor(ctx.device_ctx, ref, false, true);
    if (original_variable_conf.has_initializer()) {
      LOG(ERROR) << "id: " << this->kernel_conf().model_io_v2_conf().parallel_ctx().parallel_id()
                 << "seed id:" << seed_id_ << " seed_num: " << seed_num_;
      // std::mt19937 random_seed_gen(original_variable_conf.random_seed());
      std::seed_seq seq{original_variable_conf.random_seed()};
      std::vector<int64_t> seeds(seed_num_);
      seq.generate(seeds.begin(), seeds.end());
      int64_t seed = seeds.at(seed_id_);

      std::mt19937 random_seed_gen(seed);
      InitializeWithConfUtil::SwitchInitializeWithConf(SwitchCase(data_type),
                                                       original_variable_conf.initializer(),
                                                       random_seed_gen(), ref_accessor.host_blob());
    } else if (original_variable_conf.has_initialize_with_snapshot()) {
      const auto& snapshot_conf = original_variable_conf.initialize_with_snapshot();
      const std::string& var_lbn =
          GenLogicalBlobName(conf.variable_op_name(), original_variable_conf.out());
      const std::string key = snapshot_conf.has_key() ? snapshot_conf.key() : var_lbn;
      const Shape logical_blob_shape(original_variable_conf.shape());
      const TensorSliceView slice = GetPartSlice(this->kernel_conf());
      const SnapshotReader reader(snapshot_conf.path());
      reader.Read(key, logical_blob_shape, slice, ref_accessor.host_blob());
    } else {
      UNIMPLEMENTED();
    }
  }

  int64_t seed_id_;
  int64_t seed_num_;
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kModelInitV2Conf, ModelInitV2Kernel);

template<DeviceType device_type>
class ModelLoadV2Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelLoadV2Kernel);
  ModelLoadV2Kernel() = default;
  ~ModelLoadV2Kernel() override = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    ForwardDataContent(ctx, BnInOp2Blob);
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const ModelLoadV2OpConf& conf = this->op_conf().model_load_v2_conf();
    const Blob* path = BnInOp2Blob("path");
    Blob* ref = BnInOp2Blob("ref");
    const VariableOpConf& original_variable_conf = conf.original_variable_conf();
    const Shape logical_blob_shape(original_variable_conf.shape());
    const std::string& var_lbn =
        GenLogicalBlobName(conf.variable_op_name(), original_variable_conf.out());
    const TensorSliceView slice = GetPartSlice(this->kernel_conf());
    AutoSyncBlobAccessor<device_type> ref_accessor(ctx.device_ctx, ref, false, true);
    const std::string snapshot_path = SyncReadStringFromBlob<device_type>(ctx.device_ctx, path);
    SnapshotReader reader(snapshot_path);
    reader.Read(var_lbn, logical_blob_shape, slice, ref_accessor.host_blob());
  }
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kModelLoadV2Conf, ModelLoadV2Kernel);

template<DeviceType device_type>
class ModelSaveV2Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveV2Kernel);
  ModelSaveV2Kernel() = default;
  ~ModelSaveV2Kernel() override = default;

 private:
  void VirtualKernelInit() override {
    counter_.reset(new int64_t(0));

    const auto& parallel_desc = ParallelDesc(
        this->kernel_conf().op_attribute().parallel_conf_signature().op_parallel_conf());
    const auto& hierarchy = parallel_desc.hierarchy();
    int64_t seed_num = 1;
    const auto& map = this->kernel_conf()
                          .op_attribute()
                          .parallel_distribution_signature()
                          .bn_in_op2parallel_distribution();
    const auto it = map.find("in");
    CHECK(it != map.end());
    ParallelDistribution in_parallel_distribution = it->second;
    for (int64_t i = 0; i < hierarchy->NumAxes(); ++i) {
      SbpParallel sbp_parallel = in_parallel_distribution.sbp_parallel(i);
      CHECK(sbp_parallel.has_split_parallel() || sbp_parallel.has_broadcast_parallel());
      if (sbp_parallel.has_split_parallel()) { seed_num *= hierarchy->At(i); }
    }
    seed_num_ = seed_num;
  }

  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    ForwardDataContent(ctx, BnInOp2Blob);
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    *counter_ += 1;
    const ModelSaveV2OpConf& conf = this->op_conf().model_save_v2_conf();
    const Blob* path_blob = BnInOp2Blob("path");
    Blob* in_blob = BnInOp2Blob("in");
    const ParallelContext& parallel_ctx = this->kernel_conf().model_io_v2_conf().parallel_ctx();
    const VariableOpConf& original_variable_conf = conf.original_variable_conf();
    const Shape logical_blob_shape(original_variable_conf.shape());
    const DataType data_type = original_variable_conf.data_type();

    const auto& parallel_desc = ParallelDesc(
        this->kernel_conf().op_attribute().parallel_conf_signature().op_parallel_conf());
    const auto& hierarchy = parallel_desc.hierarchy();

    const auto& map = this->kernel_conf()
                          .op_attribute()
                          .parallel_distribution_signature()
                          .bn_in_op2parallel_distribution();
    const auto it = map.find("in");
    CHECK(it != map.end());
    ParallelDistribution in_parallel_distribution = it->second;
    for (int64_t i = 0; i < hierarchy->NumAxes(); ++i) {
      SbpParallel sbp_parallel = in_parallel_distribution.sbp_parallel(i);
      const int64_t rank_id =
          (parallel_ctx.parallel_id() % hierarchy->Count(i)) / hierarchy->Count(i + 1);
      if (sbp_parallel.has_broadcast_parallel() && rank_id != 0) { return; }
    }
    LOG(ERROR) << "seed_num_ " << seed_num_;
    const std::string snapshot_path =
        SyncReadStringFromBlob<device_type>(ctx.device_ctx, path_blob);
    LOG(ERROR) << parallel_ctx.parallel_id() << " do save " << snapshot_path;
    AutoSyncBlobAccessor<device_type> in_accessor(ctx.device_ctx, in_blob, true, false);
    SnapshotWriter writer(snapshot_path);
    const std::string var_lbn =
        GenLogicalBlobName(conf.variable_op_name(), original_variable_conf.out());
    const std::string key = GetTmpPartKey(var_lbn, parallel_ctx);
    writer.Write(key, in_accessor.host_blob());
    const int64_t parallel_num = parallel_ctx.parallel_num();
    const std::string rpc_key =
        snapshot_path + "-" + var_lbn + "-Counter-" + std::to_string(*counter_);
    int32_t counter = Global<CtrlClient>::Get()->IncreaseCount(rpc_key);

    LOG(ERROR) << parallel_ctx.parallel_id() << " rpc_key " << rpc_key << " counter " << counter;
    if (counter < seed_num_) { return; }
    TensorSliceView total_slice(logical_blob_shape);
    OnDemandHostBlob total_blob(logical_blob_shape, data_type);
    SnapshotReader reader(snapshot_path);
    FOR_RANGE(int64_t, i, 0, parallel_num) {
      bool skip = false;
      FOR_RANGE(int64_t, j, 0, hierarchy->NumAxes()) {
        SbpParallel sbp_parallel = in_parallel_distribution.sbp_parallel(j);
        const int64_t rank_id = (i % hierarchy->Count(j)) / hierarchy->Count(j + 1);
        if (sbp_parallel.has_broadcast_parallel() && rank_id != 0) {
          skip = true;
          break;
        }
      }
      if (skip) {
        LOG(ERROR) << "skip " << i;
        continue;
      }
      const TensorSliceView part_slice = GetPartSlice(this->kernel_conf(), i);
      const std::string part_key = GetTmpPartKey(var_lbn, i, parallel_num);
      LOG(ERROR) << parallel_ctx.parallel_id() << " read " << part_key;
      OnDemandHostBlob part_blob(part_slice.shape(), data_type);
      reader.Read(part_key, part_blob.blob());
      HostSliceCopy(total_blob.blob(), total_slice, part_blob.blob(), part_slice);
      SnapshotFS()->RecursivelyDeleteDir(Dirname(JoinPath(snapshot_path, part_key)));
    }
    writer.Write(var_lbn, total_blob.blob());
    Global<CtrlClient>::Get()->EraseCount(rpc_key);
  }
  std::unique_ptr<int64_t> counter_;
  int64_t seed_num_;
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kModelSaveV2Conf, ModelSaveV2Kernel);

}  // namespace oneflow
