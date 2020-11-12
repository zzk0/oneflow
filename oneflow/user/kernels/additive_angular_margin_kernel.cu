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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename K>
__device__ int64_t GetOffset(const int64_t batch_idx, const int64_t num_classes,
                             const int64_t lower_bound, const K* label) {
  const int64_t idx = label[batch_idx] - lower_bound;
  if (idx >= 0 && idx < num_classes) {
    return batch_idx * num_classes + idx;
  } else {
    return -1;
  }
}

template<typename T, typename K>
__global__ void GpuForward(const int64_t num_instances, const int64_t num_classes,
                           const int64_t lower_bound, const T cos_m, const T sin_m, const T* in,
                           const K* label, T* sin_theta_data, T* out) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    const int64_t idx = GetOffset<K>(i, num_classes, lower_bound, label);
    if (idx != -1) {
      sin_theta_data[i] = sqrt(1 - in[idx] * in[idx]);
      out[idx] = in[idx] * cos_m - sin_theta_data[i] * sin_m;
      sin_theta_data[i] = in[idx] / sin_theta_data[i];
    }
  }
}

template<typename T, typename K>
__global__ void GpuBackward(const int64_t num_instances, const int64_t num_classes,
                            const int64_t lower_bound, const T cos_m, const T sin_m,
                            const T* out_diff, const K* label, const T* sin_theta_data,
                            T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    const int64_t idx = GetOffset<K>(i, num_classes, lower_bound, label);
    if (idx != -1) { in_diff[idx] = in_diff[idx] * (1 * cos_m + sin_theta_data[i] * sin_m); }
  }
}

class AdditiveAngularMarginOpKernelState final : public user_op::OpKernelState {
 public:
  AdditiveAngularMarginOpKernelState(int64_t lower, int64_t upper) : lower_(lower), upper_(upper) {}
  ~AdditiveAngularMarginOpKernelState() override = default;

  int64_t lower() const { return lower_; }
  int64_t upper() const { return upper_; }

 private:
  const int64_t lower_;
  const int64_t upper_;
};

std::shared_ptr<user_op::OpKernelState> CreateAdditiveAngularMarginOpKernelState(
    user_op::KernelInitContext* ctx, const std::string& in_arg_name) {
  const SbpParallel& in_sbp = ctx->SbpParallel4ArgNameAndIndex(in_arg_name, 0);
  if (in_sbp.has_split_parallel() && in_sbp.split_parallel().axis() == 1
      && ctx->parallel_ctx().parallel_num() > 1) {
    CHECK(ctx->SbpParallel4ArgNameAndIndex("label", 0).has_broadcast_parallel());
    const user_op::TensorDesc* in_logical_desc =
        ctx->LogicalTensorDesc4ArgNameAndIndex(in_arg_name, 0);
    BalancedSplitter bs(ctx->Attr<int64_t>("depth"), ctx->parallel_ctx().parallel_num());
    return std::make_shared<AdditiveAngularMarginOpKernelState>(
        bs.At(ctx->parallel_ctx().parallel_id()).begin(),
        bs.At(ctx->parallel_ctx().parallel_id()).end());
  } else {
    return std::shared_ptr<user_op::OpKernelState>(nullptr);
  }
}

}  // namespace

template<typename T, typename K>
class AdditiveAngularMarginGpuKernel final : public user_op::OpKernel {
 public:
  AdditiveAngularMarginGpuKernel() = default;
  ~AdditiveAngularMarginGpuKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return CreateAdditiveAngularMarginOpKernelState(ctx, "x");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    user_op::Tensor* sin_theta_data = ctx->Tensor4ArgNameAndIndex("sin_theta_data", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float margin = ctx->Attr<float>("margin");
    int64_t lower_bound = 0;
    if (state != nullptr) {
      auto* kernel_state = dynamic_cast<AdditiveAngularMarginOpKernelState*>(state);
      CHECK_NOTNULL(kernel_state);
      CHECK_EQ(x->shape().Count(1), kernel_state->upper() - kernel_state->lower());
      lower_bound = kernel_state->lower();
    }
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), y->mut_dptr<void>(), x->dptr<void>(),
                             x->shape().elem_cnt() * GetSizeOfDataType(x->data_type()));
    Memset<DeviceType::kGPU>(
        ctx->device_ctx(), sin_theta_data->mut_dptr(), 0,
        sin_theta_data->shape().elem_cnt() * GetSizeOfDataType(sin_theta_data->data_type()));
    GpuForward<<<BlocksNum4ThreadsNum(x->shape().At(0)), kCudaThreadsNumPerBlock, 0,
                 ctx->device_ctx()->cuda_stream()>>>(
        x->shape().At(0), x->shape().Count(1), lower_bound, static_cast<T>(cos(margin)),
        static_cast<T>(sin(margin)), x->dptr<T>(), label->dptr<K>(), sin_theta_data->mut_dptr<T>(),
        y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};  // namespace oneflow

#define REGISTER_ADDITIVE_ANGULAR_MARGIN_KERNEL(in_type, indices_type)                \
  REGISTER_USER_KERNEL("additive_angular_margin")                                     \
      .SetCreateFn<AdditiveAngularMarginGpuKernel<OF_PP_PAIR_FIRST(in_type),          \
                                                  OF_PP_PAIR_FIRST(indices_type)>>()  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                             \
                       & (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(in_type)) \
                       & (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(indices_type)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ADDITIVE_ANGULAR_MARGIN_KERNEL, FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

template<typename T, typename K>
class AdditiveAngularMarginGradGpuKernel final : public user_op::OpKernel {
 public:
  AdditiveAngularMarginGradGpuKernel() = default;
  ~AdditiveAngularMarginGradGpuKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return CreateAdditiveAngularMarginOpKernelState(ctx, "dy");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    const user_op::Tensor* sin_theta_data = ctx->Tensor4ArgNameAndIndex("sin_theta_data", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const float margin = ctx->Attr<float>("margin");
    int64_t lower_bound = 0;
    if (state != nullptr) {
      auto* kernel_state = dynamic_cast<AdditiveAngularMarginOpKernelState*>(state);
      CHECK_NOTNULL(kernel_state);
      CHECK_EQ(dy->shape().Count(1), kernel_state->upper() - kernel_state->lower());
      lower_bound = kernel_state->lower();
    }
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), dx->mut_dptr<void>(), dy->dptr<void>(),
                             dy->shape().elem_cnt() * GetSizeOfDataType(dy->data_type()));
    GpuBackward<<<BlocksNum4ThreadsNum(dy->shape().At(0)), kCudaThreadsNumPerBlock, 0,
                  ctx->device_ctx()->cuda_stream()>>>(
        dy->shape().At(0), dy->shape().Count(1), lower_bound, static_cast<T>(cos(margin)),
        static_cast<T>(sin(margin)), dy->dptr<T>(), label->dptr<K>(), sin_theta_data->dptr<T>(),
        dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADDITIVE_ANGULAR_MARGIN_GRAD_KERNEL(dy_type, indices_type)              \
  REGISTER_USER_KERNEL("additive_angular_margin_grad")                                   \
      .SetCreateFn<AdditiveAngularMarginGradGpuKernel<OF_PP_PAIR_FIRST(dy_type),         \
                                                      OF_PP_PAIR_FIRST(indices_type)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                \
                       & (user_op::HobDataType("dy", 0) == OF_PP_PAIR_SECOND(dy_type))   \
                       & (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(indices_type)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ADDITIVE_ANGULAR_MARGIN_GRAD_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
