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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/random_generator.h"
#include "oneflow/core/kernel/gather_kernel_util.h"
#include "oneflow/core/kernel/unsorted_segment_sum_kernel_util.h"
#include <cub/cub.cuh>

namespace oneflow {
namespace user_op {

namespace {

template<typename K>
int64_t GetCubSortPairTempStorageSize(int64_t n) {
  size_t cub_sort_temp_store_size = 0;
  OF_CUDA_CHECK((cub::DeviceRadixSort::SortPairs<K, K>(nullptr, cub_sort_temp_store_size, nullptr,
                                                       nullptr, nullptr, nullptr, n)));
  CHECK_GE(cub_sort_temp_store_size, 0);
  CHECK_LT(cub_sort_temp_store_size, GetMaxVal<int64_t>());
  return GetCudaAlignedSize(static_cast<int64_t>(cub_sort_temp_store_size));
}

template<typename KEY, typename VAL>
void SortPairs(cudaStream_t stream, int64_t n, size_t temp_storage_bytes, const KEY* keys,
               const VAL* vals, void* tmp_storage, KEY* sorted_keys, VAL* sorted_vals) {
  OF_CUDA_CHECK((cub::DeviceRadixSort::SortPairs<KEY, VAL>(tmp_storage, temp_storage_bytes, keys,
                                                           sorted_keys, vals, sorted_vals, n, 0,
                                                           sizeof(KEY) * 8, stream)));
}

template<typename K>
class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(void* ptr, const int64_t device_num_class) : ptr_(ptr) {
    const size_t label_buffer_bytes = GetCudaAlignedSize(device_num_class * sizeof(K));
    const size_t index_buffer_bytes = GetCudaAlignedSize(device_num_class * sizeof(K));
    const size_t sorted_label_buffer_bytes = GetCudaAlignedSize(device_num_class * sizeof(K));
    const size_t sorted_index_buffer_bytes = GetCudaAlignedSize(device_num_class * sizeof(K));
    const size_t rand_value_bytes = GetCudaAlignedSize(device_num_class * sizeof(unsigned int));
    cub_tmp_storage_bytes_ = GetCubSortPairTempStorageSize<K>(device_num_class);

    label_buffer_offset_ = 0;
    index_buffer_offset_ = label_buffer_offset_ + label_buffer_bytes;
    sorted_label_buffer_offset_ = index_buffer_offset_ + index_buffer_bytes;
    sorted_index_buffer_offset_ = sorted_label_buffer_offset_ + sorted_label_buffer_offset_;
    rand_value_offset_ = sorted_index_buffer_offset_ + sorted_index_buffer_bytes;
    cub_tmp_storage_offset_ = rand_value_offset_ + rand_value_bytes;
    total_buffer_size_ = label_buffer_bytes + index_buffer_bytes + sorted_label_buffer_bytes
                         + sorted_index_buffer_bytes + rand_value_bytes + cub_tmp_storage_bytes_;
  }
  ~TmpBufferManager() = default;

  size_t GetTotalBufferSize() const { return total_buffer_size_; }
  size_t GetCubTmpStorageSize() const { return cub_tmp_storage_bytes_; }
  K* LabelBufferPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + label_buffer_offset_);
  }
  K* IndexBufferPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + index_buffer_offset_);
  }
  K* SortedLabelBufferPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + sorted_label_buffer_offset_);
  }
  K* SortedIndexBufferPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + sorted_index_buffer_offset_);
  }
  unsigned int* RandValuePtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<unsigned int*>(reinterpret_cast<char*>(ptr_) + rand_value_offset_);
  }
  K* LabelMapPtr() const { return LabelBufferPtr(); }
  void* CubTmpStoragePtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<void*>(reinterpret_cast<char*>(ptr_) + cub_tmp_storage_offset_);
  }

 private:
  size_t label_buffer_offset_;
  size_t index_buffer_offset_;
  size_t sorted_label_buffer_offset_;
  size_t sorted_index_buffer_offset_;
  size_t rand_value_offset_;
  size_t cub_tmp_storage_offset_;
  size_t cub_tmp_storage_bytes_;
  size_t total_buffer_size_;
  void* ptr_;
};

class PartialFcSampleOpKernelState final : public user_op::OpKernelState {
 public:
  PartialFcSampleOpKernelState(DeviceCtx* ctx) {
    CHECK_NOTNULL(ctx);
    OF_CURAND_CHECK(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
    OF_CURAND_CHECK(
        curandSetPseudoRandomGeneratorSeed(curand_generator_, static_cast<int64_t>(1111L)));
    OF_CURAND_CHECK(curandSetStream(curand_generator_, ctx->cuda_stream()));
  }
  ~PartialFcSampleOpKernelState() override = default;

  curandGenerator_t& gen() { return curand_generator_; }

 private:
  curandGenerator_t curand_generator_;
};

template<typename K>
__global__ void InitBuffer(const int64_t n, const unsigned int* rand_value, K* label_buffer,
                           K* index_buffer) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    label_buffer[i] = i;
    index_buffer[i] = static_cast<K>(rand_value[i] % n);
  }
}

template<typename K>
__global__ void IndexSetPos(const int64_t n, const int64_t offset, const int64_t num_classes,
                            const K* labels, K* index_buffer) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    K label = labels[i] - offset;
    printf("index_buffer[%d] offset %d \n", labels[i], offset);
    if (label >= 0 && label < num_classes) {
      index_buffer[label] = -1;
    }
  }
}

template<typename K>
__global__ void GetLabelMap(const int64_t n, const int64_t map_offset, const K* label,
                            K* label_map) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    K cur_label = label[i];
    label_map[cur_label] = map_offset + i;
  }
}

template<typename K>
__global__ void GetSampleLabel(const int64_t n, const int64_t offset, const K* label,
                               K* sample_label) {
  CUDA_1D_KERNEL_LOOP(i, n) { sample_label[i] = label[i] + offset; }
}

}  // namespace

template<typename K>
class PartialFcSampleGpuKernel final : public user_op::OpKernel {
 public:
  PartialFcSampleGpuKernel() = default;
  ~PartialFcSampleGpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<PartialFcSampleOpKernelState>(ctx->device_ctx());
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    user_op::Tensor* maped_label = ctx->Tensor4ArgNameAndIndex("maped_label", 0);
    user_op::Tensor* sampled_label = ctx->Tensor4ArgNameAndIndex("sampled_index", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const int64_t batch_size = label->shape().At(0);
    const int64_t num_classes = ctx->Attr<int64_t>("num_classes");
    const int64_t lower_bound = ctx->Attr<int64_t>("class_offset");
    const int64_t num_sample = ctx->Attr<int64_t>("num_sample");
    TmpBufferManager<K> buffer_manager(tmp_buffer->mut_dptr(), num_classes);
    auto* kernel_state = dynamic_cast<PartialFcSampleOpKernelState*>(state);
    CHECK_NOTNULL(kernel_state);
    OF_CURAND_CHECK(
        curandGenerate(kernel_state->gen(), buffer_manager.RandValuePtr(), num_classes));
    InitBuffer<<<BlocksNum4ThreadsNum(num_classes), kCudaThreadsNumPerBlock, 0,
                 ctx->device_ctx()->cuda_stream()>>>(num_classes, buffer_manager.RandValuePtr(),
                                                     buffer_manager.LabelBufferPtr(),
                                                     buffer_manager.IndexBufferPtr());

    IndexSetPos<<<BlocksNum4ThreadsNum(batch_size), kCudaThreadsNumPerBlock, 0,
                  ctx->device_ctx()->cuda_stream()>>>(
        batch_size, lower_bound, num_classes, label->dptr<K>(), buffer_manager.IndexBufferPtr());

    SortPairs<K, K>(ctx->device_ctx()->cuda_stream(), num_classes,
                    buffer_manager.GetCubTmpStorageSize(), buffer_manager.IndexBufferPtr(),
                    buffer_manager.LabelBufferPtr(), buffer_manager.CubTmpStoragePtr(),
                    buffer_manager.SortedIndexBufferPtr(), buffer_manager.SortedLabelBufferPtr());
    // check num_sample > num_pos
    // get sampled_label
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), sampled_label->mut_dptr<void>(),
                             buffer_manager.SortedLabelBufferPtr(),
                             num_sample * GetSizeOfDataType(sampled_label->data_type()));
    // get LabelMap
    const int64_t map_offset = ctx->Attr<int64_t>("sample_offset");
    GetLabelMap<<<BlocksNum4ThreadsNum(num_sample), kCudaThreadsNumPerBlock, 0,
                  ctx->device_ctx()->cuda_stream()>>>(num_sample, map_offset,
                                                      buffer_manager.SortedLabelBufferPtr(),
                                                      buffer_manager.LabelMapPtr());
    Memset<DeviceType::kGPU>(ctx->device_ctx(), maped_label->mut_dptr(), 0,
                             maped_label->shape().elem_cnt() * sizeof(K));
    GatherKernelUtilImpl<DeviceType::kGPU, K, K>::Forward(
        ctx->device_ctx(), label->dptr<K>(), batch_size, buffer_manager.LabelMapPtr(),
        Shape({1, num_classes, 1}), maped_label->mut_dptr<K>(), lower_bound);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_PARTIAL_FC_SAMPLE_GPU_KERNEL(dtype)                                         \
  REGISTER_USER_KERNEL("partial_fc_sample")                                                       \
      .SetCreateFn<                                                                               \
          PartialFcSampleGpuKernel<dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                         \
                       & (user_op::HobDataType("label", 0) == GetDataType<dtype>::value))     \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                                \
        const int64_t num_classes = ctx->Attr<int64_t>("num_classes");                            \
        TmpBufferManager<dtype> buffer_manager(nullptr, num_classes);      \
        return buffer_manager.GetTotalBufferSize();                                               \
      });

REGISTER_PARTIAL_FC_SAMPLE_GPU_KERNEL(int32_t)

}  // namespace user_op
}  // namespace oneflow
