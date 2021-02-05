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
#include "oneflow/core/kernel/gather_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace user_op {

namespace {

Shape GetFlatShape(const ShapeView& shape, int64_t axis) {
  return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

class GatherOpKernelState final : public user_op::OpKernelState {
 public:
  GatherOpKernelState(int64_t lower, int64_t upper) : lower_(lower), upper_(upper) {}
  ~GatherOpKernelState() override = default;

  int64_t lower() const { return lower_; }
  int64_t upper() const { return upper_; }

 private:
  const int64_t lower_;
  const int64_t upper_;
};

}  // namespace

template<DeviceType device_type, typename T, typename K>
class GatherKernel final : public user_op::OpKernel {
 public:
  GatherKernel() = default;
  ~GatherKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto axis = ctx->Attr<int64_t>("axis");
    const ParallelDistribution& in_parallel_distribution =
        ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
    const ParallelDistribution& indices_parallel_distribution =
        ctx->ParallelDistribution4ArgNameAndIndex("indices", 0);
    const ParallelDistribution& out_parallel_distribution =
        ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
    LOG(ERROR) << "in_parallel_distribution:\n " << in_parallel_distribution.DebugString();
    const Shape& in_parallel_hierarchy = ctx->ParallelHierarchy();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    const TensorDesc* in_logical_desc = ctx->LogicalTensorDesc4ArgNameAndIndex("in", 0);
    const int64_t gather_dim_size = in_logical_desc->shape().At(axis);
    if (in_parallel_hierarchy.NumAxes() == 1) {
      const SbpParallel& in_sbp = in_parallel_distribution.sbp_parallel(0);
      if (in_sbp.has_split_parallel() && in_sbp.split_parallel().axis() == axis) {
        CHECK(indices_parallel_distribution.sbp_parallel(0).has_broadcast_parallel());
        CHECK(out_parallel_distribution.sbp_parallel(0).has_partial_sum_parallel());
        BalancedSplitter bs(gather_dim_size, ctx->parallel_ctx().parallel_num());
        return std::make_shared<GatherOpKernelState>(bs.At(parallel_id).begin(),
                                                     bs.At(parallel_id).end());
      } else {
        return std::shared_ptr<OpKernelState>(nullptr);
      }
    } else {
      CHECK_EQ(in_parallel_distribution.sbp_parallel_size(), 2);
      CHECK_EQ(indices_parallel_distribution.sbp_parallel_size(), 2);
      CHECK_EQ(out_parallel_distribution.sbp_parallel_size(), 2);
      const SbpParallel& in_0_sbp = in_parallel_distribution.sbp_parallel(0);
      const SbpParallel& in_1_sbp = in_parallel_distribution.sbp_parallel(1);

      const int64_t parallel_rank_0 = parallel_id / in_parallel_hierarchy.At(1);
      const int64_t parallel_rank_1 = parallel_id % in_parallel_hierarchy.At(1);

      const TensorDesc* in_logical_desc = ctx->LogicalTensorDesc4ArgNameAndIndex("in", 0);
      const int64_t gather_dim_size = in_logical_desc->shape().At(axis);
      const bool is_sbp_0_split_axis =
          (in_0_sbp.has_split_parallel() && in_0_sbp.split_parallel().axis() == axis);
      const bool is_sbp_1_split_axis =
          (in_1_sbp.has_split_parallel() && in_1_sbp.split_parallel().axis() == axis);
      if (is_sbp_0_split_axis) {
        CHECK(indices_parallel_distribution.sbp_parallel(0).has_broadcast_parallel());
        CHECK(out_parallel_distribution.sbp_parallel(0).has_partial_sum_parallel());
      }
      if (is_sbp_1_split_axis) {
        CHECK(indices_parallel_distribution.sbp_parallel(1).has_broadcast_parallel());
        CHECK(out_parallel_distribution.sbp_parallel(1).has_partial_sum_parallel());
      }
      if (is_sbp_0_split_axis && is_sbp_1_split_axis) {
        BalancedSplitter bs(gather_dim_size, ctx->parallel_ctx().parallel_num());
        return std::make_shared<GatherOpKernelState>(bs.At(parallel_id).begin(),
                                                     bs.At(parallel_id).end());
      } else if (is_sbp_0_split_axis && !is_sbp_1_split_axis) {
        BalancedSplitter bs(gather_dim_size, in_parallel_hierarchy.At(0));
        return std::make_shared<GatherOpKernelState>(bs.At(parallel_rank_0).begin(),
                                                     bs.At(parallel_rank_0).end());
      } else if (is_sbp_1_split_axis && !is_sbp_0_split_axis) {
        BalancedSplitter bs(gather_dim_size, in_parallel_hierarchy.At(1));
        return std::make_shared<GatherOpKernelState>(bs.At(parallel_rank_1).begin(),
                                                     bs.At(parallel_rank_1).end());
      } else {
        return std::shared_ptr<OpKernelState>(nullptr);
      }
    }
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    const int64_t axis = ctx->Attr<int64_t>("axis");
    const int64_t num_indices = indices->shape().elem_cnt();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    int64_t offset = 0;
    if (state != nullptr) {
      auto* gather_state = dynamic_cast<GatherOpKernelState*>(state);
      CHECK_NOTNULL(gather_state);
      CHECK_EQ(in->shape().At(axis), gather_state->upper() - gather_state->lower());
      offset = gather_state->lower();
      LOG(ERROR) << "id: " << ctx->parallel_ctx().parallel_id()
                 << "lower: " << gather_state->lower() << "upper" << gather_state->upper();
    }

    GatherKernelUtilImpl<device_type, T, K>::Forward(
        ctx->device_ctx(), indices->dptr<K>(), num_indices, in->dptr<T>(),
        GetFlatShape(in->shape(), axis), out->mut_dptr<T>(), offset);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GATHER_KERNEL(device, in_type, indices_type)                                \
  REGISTER_USER_KERNEL("gather")                                                             \
      .SetCreateFn<                                                                          \
          GatherKernel<device, OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indices_type)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                   \
                       & (user_op::HobDataType("in", 0) == OF_PP_PAIR_SECOND(in_type))       \
                       & (user_op::HobDataType("indices", 0) == OF_PP_PAIR_SECOND(indices_type)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_GATHER_KERNEL, DEVICE_TYPE_SEQ, GATHER_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

}  // namespace user_op

}  // namespace oneflow
