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

#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_context.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/chain_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/collective_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/slice_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/naive_b2b_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/naive_b2p_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/b21_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/one_to_one_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/slice_boxing_task_node.h"

namespace oneflow {
namespace {

void ParallelhierarchyReduce(const Shape& parallel_hierarchy,
                             const ParallelDistribution& parallel_distribution,
                             Shape* return_parallel_hierarchy,
                             ParallelDistribution* return_parallel_distribution) {
  DimVector reduced_parallel_hierarchy;
  reduced_parallel_hierarchy.push_back(parallel_hierarchy.At(0));
  *return_parallel_distribution->add_sbp_parallel() = parallel_distribution.sbp_parallel(0);
  FOR_RANGE(int64_t, i, 1, parallel_hierarchy.NumAxes()) {
    if (parallel_distribution.sbp_parallel(i) == parallel_distribution.sbp_parallel(i - 1)) {
      // contiguous, can reduce
      reduced_parallel_hierarchy.back() *= parallel_hierarchy.At(i);
    } else {
      reduced_parallel_hierarchy.push_back(parallel_hierarchy.At(i));
      *return_parallel_distribution->add_sbp_parallel() = parallel_distribution.sbp_parallel(i);
    }
  }
  *return_parallel_hierarchy = Shape(reduced_parallel_hierarchy);
}

void CollaborativeParallelhierarchyReduce(
    const Shape& src_parallel_hierarchy, const ParallelDistribution& src_parallel_distribution,
    const Shape& dst_parallel_hierarchy, const ParallelDistribution& dst_parallel_distribution,
    Shape* src_return_parallel_hierarchy, ParallelDistribution* src_return_parallel_distribution,
    Shape* dst_return_parallel_hierarchy, ParallelDistribution* dst_return_parallel_distribution) {
  CHECK_EQ(src_parallel_hierarchy.NumAxes(), dst_parallel_hierarchy.NumAxes());
  DimVector src_reduced_parallel_hierarchy;
  src_reduced_parallel_hierarchy.push_back(src_parallel_hierarchy.At(0));
  *src_return_parallel_distribution->add_sbp_parallel() = src_parallel_distribution.sbp_parallel(0);

  DimVector dst_reduced_parallel_hierarchy;
  dst_reduced_parallel_hierarchy.push_back(dst_parallel_hierarchy.At(0));
  *dst_return_parallel_distribution->add_sbp_parallel() = dst_parallel_distribution.sbp_parallel(0);

  FOR_RANGE(int64_t, i, 1, src_parallel_hierarchy.NumAxes()) {
    if ((src_parallel_distribution.sbp_parallel(i) == src_parallel_distribution.sbp_parallel(i - 1))
        && (dst_parallel_distribution.sbp_parallel(i)
            == dst_parallel_distribution.sbp_parallel(i - 1))) {
      // contiguous, can reduce
      src_reduced_parallel_hierarchy.back() *= src_parallel_hierarchy.At(i);
      dst_reduced_parallel_hierarchy.back() *= dst_parallel_hierarchy.At(i);
    } else {
      src_reduced_parallel_hierarchy.push_back(src_parallel_hierarchy.At(i));
      *src_return_parallel_distribution->add_sbp_parallel() =
          src_parallel_distribution.sbp_parallel(i);

      dst_reduced_parallel_hierarchy.push_back(dst_parallel_hierarchy.At(i));
      *dst_return_parallel_distribution->add_sbp_parallel() =
          dst_parallel_distribution.sbp_parallel(i);
    }
  }
  *src_return_parallel_hierarchy = Shape(src_reduced_parallel_hierarchy);
  *dst_return_parallel_hierarchy = Shape(dst_reduced_parallel_hierarchy);
}

void InOutParallelhierarchyReduce(const Shape& in_parallel_hierarchy,
                                  const Shape& out_parallel_hierarchy,
                                  const ParallelDistribution& in_parallel_distribution,
                                  const ParallelDistribution& out_parallel_distribution,
                                  Shape* return_in_parallel_hierarchy,
                                  Shape* return_out_parallel_hierarchy,
                                  ParallelDistribution* return_in_parallel_distribution,
                                  ParallelDistribution* return_out_parallel_distribution) {
  if (in_parallel_hierarchy.NumAxes() == 1 && out_parallel_hierarchy.NumAxes() == 1) {
    LOG(INFO) << "1D hierarchy";
    *return_in_parallel_distribution->add_sbp_parallel() = in_parallel_distribution.sbp_parallel(0);
    *return_out_parallel_distribution->add_sbp_parallel() =
        out_parallel_distribution.sbp_parallel(0);
    *return_in_parallel_hierarchy = Shape(in_parallel_hierarchy);
    *return_out_parallel_hierarchy = Shape(out_parallel_hierarchy);
  } else if (in_parallel_hierarchy.NumAxes() != out_parallel_hierarchy.NumAxes()) {
    LOG(INFO) << "ParallelhierarchyReduce";
    ParallelhierarchyReduce(in_parallel_hierarchy, in_parallel_distribution,
                            return_in_parallel_hierarchy, return_in_parallel_distribution);
    ParallelhierarchyReduce(out_parallel_hierarchy, out_parallel_distribution,
                            return_out_parallel_hierarchy, return_out_parallel_distribution);
  } else {
    LOG(INFO) << "CollaborativeParallelhierarchyReduce";
    // reduce when both src and dst can reduce.
    CollaborativeParallelhierarchyReduce(
        in_parallel_hierarchy, in_parallel_distribution, out_parallel_hierarchy,
        out_parallel_distribution, return_in_parallel_hierarchy, return_in_parallel_distribution,
        return_out_parallel_hierarchy, return_out_parallel_distribution);
  }
}

Maybe<SubTskGphBuilderStatus> BuildFirstAxisParallelDistributionChangeSubGph(
    SubTskGphBuilderCtx* ctx, const std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder,
    const std::vector<TaskNode*>& sorted_in_tasks, std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const Shape& in_parallel_hierarchy,
    const Shape& out_parallel_hierarchy, const ParallelDistribution& in_parallel_distribution,
    const ParallelDistribution& out_parallel_distribution, const Shape& time_shape) {
  CHECK_EQ_OR_RETURN(in_parallel_hierarchy.NumAxes(), 2);
  CHECK_EQ_OR_RETURN(in_parallel_hierarchy, out_parallel_hierarchy);
  CHECK_OR_RETURN(in_parallel_distribution.sbp_parallel(1)
                  == out_parallel_distribution.sbp_parallel(1));
  CHECK_OR_RETURN(in_parallel_distribution.sbp_parallel(0)
                  != out_parallel_distribution.sbp_parallel(0));

  LOG(INFO) << "2D sbp axis 0 BuildFirstAxisParallelDistributionChangeSubGph";
  std::vector<SubTskGphBuilderStatus> status;
  std::vector<std::vector<TaskNode*>> out_nodes(in_parallel_hierarchy.At(0));
  FOR_RANGE(int64_t, i, 0, in_parallel_hierarchy.At(1)) {
    std::vector<TaskNode*> in_tasks;
    std::vector<TaskNode*> out_tasks;
    std::vector<std::vector<TaskNode*>> ctrl_tasks;
    ctrl_tasks.resize(in_parallel_hierarchy.At(0));
    ParallelConf in_parallel_conf;
    ParallelConf out_parallel_conf;
    in_parallel_conf.set_device_tag(in_parallel_desc.device_tag());
    out_parallel_conf.set_device_tag(out_parallel_desc.device_tag());
    FOR_RANGE(int64_t, j, 0, in_parallel_hierarchy.At(0)) {
      const int64_t parallel_id = j * out_parallel_hierarchy.At(1) + i;
      in_tasks.push_back(sorted_in_tasks.at(parallel_id));
      std::string in_machine_id =
          std::to_string(CHECK_JUST(in_parallel_desc.MachineId4ParallelId(parallel_id)));
      std::string in_device_id =
          std::to_string(CHECK_JUST(in_parallel_desc.DeviceId4ParallelId(parallel_id)));
      in_parallel_conf.add_device_name(in_machine_id + ":" + in_device_id);
      std::string out_machine_id =
          std::to_string(CHECK_JUST(out_parallel_desc.MachineId4ParallelId(parallel_id)));
      std::string out_device_id =
          std::to_string(CHECK_JUST(out_parallel_desc.DeviceId4ParallelId(parallel_id)));
      out_parallel_conf.add_device_name(out_machine_id + ":" + out_device_id);
    }
    LOG(INFO) << i << " in_parallel_conf \n" << in_parallel_conf.DebugString();
    LOG(INFO) << i << " out_parallel_conf \n" << out_parallel_conf.DebugString();
    ParallelDesc sub_builder_in_parallel_desc(in_parallel_conf);
    ParallelDesc sub_builder_out_parallel_desc(out_parallel_conf);
    const SbpParallel& in_sbp_parallel = in_parallel_distribution.sbp_parallel(0);
    const SbpParallel& out_sbp_parallel = out_parallel_distribution.sbp_parallel(0);
    DimVector dim_vec = logical_blob_desc.shape().dim_vec();
    if (in_parallel_distribution.sbp_parallel(1).has_split_parallel()) {
      const int64_t axis = in_parallel_distribution.sbp_parallel(1).split_parallel().axis();
      dim_vec.at(axis) /= in_parallel_hierarchy.At(1);
    }
    BlobDesc new_blob_desc(Shape(dim_vec), logical_blob_desc.data_type());
    Maybe<SubTskGphBuilderStatus> boxing_builder_status = TRY(
        sub_tsk_gph_builder->Build(ctx, in_tasks, &out_tasks, &ctrl_tasks,
                                   sub_builder_in_parallel_desc, sub_builder_out_parallel_desc, lbi,
                                   new_blob_desc, in_sbp_parallel, out_sbp_parallel, time_shape));
    LOG(INFO) << " builder_name: " << CHECK_JUST(boxing_builder_status)->builder_name() << "   "
              << CHECK_JUST(boxing_builder_status)->comment();

    status.push_back(*CHECK_JUST(boxing_builder_status));

    CHECK_EQ_OR_RETURN(out_tasks.size(), in_parallel_hierarchy.At(0));
    FOR_RANGE(int64_t, j, 0, in_parallel_hierarchy.At(0)) {
      const int64_t parallel_id = j * out_parallel_hierarchy.At(1) + i;
      out_nodes.at(j).push_back(out_tasks.at(j));
      for (TaskNode* ctrl_node : ctrl_tasks.at(j)) {
        sorted_ctrl_tasks->at(parallel_id).push_back(ctrl_node);
      }
    }
  }
  FOR_RANGE(int64_t, i, 0, in_parallel_hierarchy.At(0)) {
    FOR_RANGE(int64_t, j, 0, in_parallel_hierarchy.At(1)) {
      sorted_out_tasks->push_back(out_nodes.at(i).at(j));
    }
  }
  Maybe<SubTskGphBuilderStatus> composed_status = MakeComposedSubTskGphBuilderStatus(status);
  return composed_status;
}

Maybe<SubTskGphBuilderStatus> BuildLastAxisParallelDistributionChangeSubGph(
    SubTskGphBuilderCtx* ctx, const std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder,
    const std::vector<TaskNode*>& sorted_in_tasks, std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const Shape& in_parallel_hierarchy,
    const Shape& out_parallel_hierarchy, const ParallelDistribution& in_parallel_distribution,
    const ParallelDistribution& out_parallel_distribution, const Shape& time_shape) {
  CHECK_EQ_OR_RETURN(in_parallel_hierarchy.NumAxes(), 2);
  CHECK_EQ_OR_RETURN(in_parallel_hierarchy, out_parallel_hierarchy);
  CHECK_OR_RETURN(in_parallel_distribution.sbp_parallel(1)
                  != out_parallel_distribution.sbp_parallel(1));
  CHECK_OR_RETURN(in_parallel_distribution.sbp_parallel(0)
                  == out_parallel_distribution.sbp_parallel(0));

  LOG(INFO) << "2D sbp axis 1 BuildFirstAxisParallelDistributionChangeSubGph";
  std::vector<SubTskGphBuilderStatus> status;
  FOR_RANGE(int64_t, i, 0, in_parallel_hierarchy.At(0)) {
    std::vector<TaskNode*> in_tasks;
    std::vector<TaskNode*> out_tasks;
    std::vector<std::vector<TaskNode*>> ctrl_tasks;
    ctrl_tasks.resize(in_parallel_hierarchy.At(1));
    ParallelConf in_parallel_conf;
    ParallelConf out_parallel_conf;
    in_parallel_conf.set_device_tag(in_parallel_desc.device_tag());
    out_parallel_conf.set_device_tag(out_parallel_desc.device_tag());
    FOR_RANGE(int64_t, j, 0, in_parallel_hierarchy.At(1)) {
      const int64_t parallel_id = i * out_parallel_hierarchy.At(1) + j;
      in_tasks.push_back(sorted_in_tasks.at(parallel_id));
      std::string in_machine_id =
          std::to_string(CHECK_JUST(in_parallel_desc.MachineId4ParallelId(parallel_id)));
      std::string in_device_id =
          std::to_string(CHECK_JUST(in_parallel_desc.DeviceId4ParallelId(parallel_id)));
      in_parallel_conf.add_device_name(in_machine_id + ":" + in_device_id);
      std::string out_machine_id =
          std::to_string(CHECK_JUST(out_parallel_desc.MachineId4ParallelId(parallel_id)));
      std::string out_device_id =
          std::to_string(CHECK_JUST(out_parallel_desc.DeviceId4ParallelId(parallel_id)));
      out_parallel_conf.add_device_name(out_machine_id + ":" + out_device_id);
    }
    ParallelDesc in_parallel_desc(in_parallel_conf);
    ParallelDesc out_parallel_desc(out_parallel_conf);
    const SbpParallel& in_sbp_parallel = in_parallel_distribution.sbp_parallel(1);
    const SbpParallel& out_sbp_parallel = out_parallel_distribution.sbp_parallel(1);
    DimVector dim_vec = logical_blob_desc.shape().dim_vec();
    if (in_parallel_distribution.sbp_parallel(0).has_split_parallel()) {
      const int64_t axis = in_parallel_distribution.sbp_parallel(0).split_parallel().axis();
      dim_vec.at(axis) /= in_parallel_hierarchy.At(0);
    }
    BlobDesc new_blob_desc(Shape(dim_vec), logical_blob_desc.data_type());
    Maybe<SubTskGphBuilderStatus> boxing_builder_status = TRY(sub_tsk_gph_builder->Build(
        ctx, in_tasks, &out_tasks, &ctrl_tasks, in_parallel_desc, out_parallel_desc, lbi,
        new_blob_desc, in_sbp_parallel, out_sbp_parallel, time_shape));
    LOG(INFO) << " builder_name: " << CHECK_JUST(boxing_builder_status)->builder_name() << "   "
              << CHECK_JUST(boxing_builder_status)->comment();

    status.push_back(*CHECK_JUST(boxing_builder_status));

    FOR_RANGE(int64_t, j, 0, in_parallel_hierarchy.At(1)) {
      const int64_t parallel_id = i * out_parallel_hierarchy.At(1) + j;
      sorted_out_tasks->push_back(out_tasks.at(j));
      for (TaskNode* ctrl_node : ctrl_tasks.at(j)) {
        sorted_ctrl_tasks->at(parallel_id).push_back(ctrl_node);
      }
    }
  }
  Maybe<SubTskGphBuilderStatus> composed_status = MakeComposedSubTskGphBuilderStatus(status);
  return composed_status;
}

Maybe<SubTskGphBuilderStatus> Build2DSliceBoxingSubTskGph(
    SubTskGphBuilderCtx* ctx, const std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder,
    const std::vector<TaskNode*>& sorted_in_tasks, std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const Shape& in_parallel_hierarchy,
    const Shape& out_parallel_hierarchy, const ParallelDistribution& in_parallel_distribution,
    const ParallelDistribution& out_parallel_distribution, const Shape& time_shape) {
  // CHECK_OR_RETURN(in_parallel_distribution.sbp_parallel(0).has_split_parallel());
  // CHECK_OR_RETURN(in_parallel_distribution.sbp_parallel(1).has_split_parallel());
  // CHECK_OR_RETURN(out_parallel_distribution.sbp_parallel(0).has_split_parallel());
  // CHECK_OR_RETURN(out_parallel_distribution.sbp_parallel(1).has_split_parallel());
  const auto NewEdge = [&ctx]() -> TaskEdge* { return ctx->task_graph()->NewEdge(); };
  const auto GetBoxingGpuThrdId = [](const int64_t dev_id, CudaWorkType work_type) -> int64_t {
#ifdef WITH_CUDA
    if (work_type == CudaWorkType::kCopyH2D) {
      return Global<IDMgr>::Get()->GetGpuH2DThrdId(dev_id);
    } else if (work_type == CudaWorkType::kCopyD2H) {
      return Global<IDMgr>::Get()->GetGpuD2HThrdId(dev_id);
    } else {
      return Global<IDMgr>::Get()->GetGpuMixThrdId(dev_id);
    }
#else
    UNIMPLEMENTED();
#endif
  };
  const auto CreateBoxingNode121 = [&ctx, &lbi, &GetBoxingGpuThrdId](
                                       const ParallelDesc& pd, const int64_t parallel_id,
                                       const TensorSliceView& slice,
                                       SliceBoxingTaskMode mode) -> SliceBoxingTaskNode* {
    SliceBoxingTaskNode* node = ctx->task_graph()->NewNode<SliceBoxingTaskNode>();
    const int64_t machine_id = CHECK_JUST(pd.MachineId4ParallelId(parallel_id));
    int64_t thrd_id = -1;
    if (pd.device_type() == DeviceType::kCPU) {
      thrd_id = Global<IDMgr>::Get()->PickCpuThrdIdEvenly(machine_id);
    } else if (pd.device_type() == DeviceType::kGPU) {
#ifdef WITH_CUDA
      thrd_id = GetBoxingGpuThrdId(CHECK_JUST(pd.DeviceId4ParallelId(parallel_id)),
                                   CudaWorkType::kCopyH2D);
#else
      UNIMPLEMENTED();
#endif
    } else {
      UNIMPLEMENTED();
    }
    node->Init(lbi, slice, mode, machine_id, thrd_id);
    return node;
  };
  LOG(INFO) << "Build2DSliceBoxingSubTskGph";
  const std::vector<TensorSliceView> in_slices = SubTskGphBuilderUtil::GetTensor2DSliceView(
      in_parallel_hierarchy, in_parallel_distribution, logical_blob_desc);
  const std::vector<TensorSliceView> out_slices = SubTskGphBuilderUtil::GetTensor2DSliceView(
      out_parallel_hierarchy, out_parallel_distribution, logical_blob_desc);
  const int64_t in_parallel_num = in_parallel_desc.parallel_num();
  const int64_t out_parallel_num = out_parallel_desc.parallel_num();
  FOR_RANGE(int64_t, out_id, 0, out_parallel_num) {
    const TensorSliceView& out_slice = out_slices.at(out_id);
    SliceBoxingTaskNode* out_node =
        CreateBoxingNode121(out_parallel_desc, out_id, out_slice, kSliceBoxingTaskModeCopy);
    FOR_RANGE(int64_t, in_id, 0, in_parallel_num) {
      const TensorSliceView& in_slice = in_slices.at(in_id);
      const TensorSliceView& intersection = out_slice.Intersect(in_slice);
      if (intersection.IsEmpty()) { continue; }
      TaskNode* in_node = sorted_in_tasks.at(in_id);
      SliceBoxingTaskNode* in_copy_node =
          CreateBoxingNode121(in_parallel_desc, in_id, intersection, kSliceBoxingTaskModeCopy);
      in_copy_node->ConnectToSrcNodeWithSlice(in_node, NewEdge(), in_slice);
      TaskNode* proxy_node = ctx->GetProxyNode(in_copy_node, in_copy_node->MemZoneId121(),
                                               out_node->machine_id(), out_node->MemZoneId121());
      SliceBoxingTaskNode* out_copy_node =
          CreateBoxingNode121(out_parallel_desc, out_id, intersection, kSliceBoxingTaskModeCopy);
      out_copy_node->ConnectToSrcNodeWithSlice(proxy_node, NewEdge(), intersection);
      out_node->ConnectToSrcNodeWithSlice(out_copy_node, NewEdge(), intersection);
    }
    sorted_out_tasks->push_back(out_node);
  }
  return TRY(BuildSubTskGphBuilderStatus("2DSliceBoxingSubTskGphBuilder", "S2S"));
}

Maybe<SubTskGphBuilderStatus> BuildSameParallelHierarchySubTskGph(
    SubTskGphBuilderCtx* ctx, const std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder,
    const std::vector<TaskNode*>& sorted_in_tasks, std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const Shape& in_parallel_hierarchy,
    const Shape& out_parallel_hierarchy, const ParallelDistribution& in_parallel_distribution,
    const ParallelDistribution& out_parallel_distribution, const Shape& time_shape) {
  CHECK_EQ_OR_RETURN(in_parallel_hierarchy, out_parallel_hierarchy);
  if (in_parallel_hierarchy.NumAxes() == 2) {
    if (!(in_parallel_distribution.sbp_parallel(0).has_partial_sum_parallel()
          || in_parallel_distribution.sbp_parallel(1).has_partial_sum_parallel())) {
      return Build2DSliceBoxingSubTskGph(
          ctx, sub_tsk_gph_builder, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
          in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc, in_parallel_hierarchy,
          out_parallel_hierarchy, in_parallel_distribution, out_parallel_distribution, time_shape);
    } else if (in_parallel_distribution.sbp_parallel(0)
               == out_parallel_distribution.sbp_parallel(0)) {
      return BuildLastAxisParallelDistributionChangeSubGph(
          ctx, sub_tsk_gph_builder, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
          in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc, in_parallel_hierarchy,
          out_parallel_hierarchy, in_parallel_distribution, out_parallel_distribution, time_shape);
    } else if (in_parallel_distribution.sbp_parallel(1)
               == out_parallel_distribution.sbp_parallel(1)) {
      return BuildFirstAxisParallelDistributionChangeSubGph(
          ctx, sub_tsk_gph_builder, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
          in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc, in_parallel_hierarchy,
          out_parallel_hierarchy, in_parallel_distribution, out_parallel_distribution, time_shape);
    } else {
      // s0, s1->s1, s0
      // s0,s1->s0,s0->s1,s0
      // fist axis 1 , then axis 0
      std::vector<SubTskGphBuilderStatus> status;
      std::vector<TaskNode*> out_tasks;
      std::vector<std::vector<TaskNode*>> ctrl_tasks;
      ParallelDistribution intermediate_parallel_distribution;
      *intermediate_parallel_distribution.add_sbp_parallel() =
          in_parallel_distribution.sbp_parallel(0);
      *intermediate_parallel_distribution.add_sbp_parallel() =
          out_parallel_distribution.sbp_parallel(1);
      Maybe<SubTskGphBuilderStatus> first_status = BuildLastAxisParallelDistributionChangeSubGph(
          ctx, sub_tsk_gph_builder, sorted_in_tasks, &out_tasks, &ctrl_tasks, in_parallel_desc,
          out_parallel_desc, lbi, logical_blob_desc, in_parallel_hierarchy, out_parallel_hierarchy,
          in_parallel_distribution, intermediate_parallel_distribution, time_shape);
      status.push_back(*CHECK_JUST(first_status));
      // todo: process ctrl
      Maybe<SubTskGphBuilderStatus> second_status = BuildFirstAxisParallelDistributionChangeSubGph(
          ctx, sub_tsk_gph_builder, out_tasks, sorted_out_tasks, sorted_ctrl_tasks,
          in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc, in_parallel_hierarchy,
          out_parallel_hierarchy, intermediate_parallel_distribution, out_parallel_distribution,
          time_shape);
      status.push_back(*CHECK_JUST(second_status));
      Maybe<SubTskGphBuilderStatus> composed_status = MakeComposedSubTskGphBuilderStatus(status);
      return composed_status;
    }
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

Maybe<SubTskGphBuilderStatus> HierarchicalSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const Shape& in_parallel_hierarchy,
    const Shape& out_parallel_hierarchy, const ParallelDistribution& in_parallel_distribution,
    const ParallelDistribution& out_parallel_distribution, const Shape& time_shape) const {
  std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder_;
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
  builders.emplace_back(new OneToOneSubTskGphBuilder());
  builders.emplace_back(new B21SubTskGphBuilder());
  builders.emplace_back(new CollectiveBoxingSubTskGphBuilder());
  builders.emplace_back(new SliceBoxingSubTskGphBuilder());
  builders.emplace_back(new NaiveB2BSubTskGphBuilder());
  builders.emplace_back(new NaiveB2PSubTskGphBuilder());
  sub_tsk_gph_builder_.reset(new ChainSubTskGphBuilder(builders));

  Shape reduced_in_parallel_hierarchy;
  Shape reduced_out_parallel_hierarchy;
  ParallelDistribution reduced_in_parallel_distribution;
  ParallelDistribution reduced_out_parallel_distribution;
  LOG(INFO) << "before reduce in_parallel_distribution \n"
            << in_parallel_distribution.DebugString();
  LOG(INFO) << "before reduce out_parallel_distribution \n"
            << out_parallel_distribution.DebugString();
  InOutParallelhierarchyReduce(
      in_parallel_hierarchy, out_parallel_hierarchy, in_parallel_distribution,
      out_parallel_distribution, &reduced_in_parallel_hierarchy, &reduced_out_parallel_hierarchy,
      &reduced_in_parallel_distribution, &reduced_out_parallel_distribution);
  CHECK_LE_OR_RETURN(reduced_in_parallel_hierarchy.NumAxes(), 2);
  CHECK_LE_OR_RETURN(reduced_out_parallel_hierarchy.NumAxes(), 2);

  LOG(INFO) << "reduced_in_parallel_distribution \n"
            << reduced_in_parallel_distribution.DebugString();
  LOG(INFO) << "reduced_out_parallel_distribution \n"
            << reduced_out_parallel_distribution.DebugString();
  if (reduced_in_parallel_hierarchy.NumAxes() == 1
      && reduced_out_parallel_hierarchy.NumAxes() == 1) {
    //(3)[s0]->(5)[B]
    LOG(INFO) << "1D sbp condition: reduced_in_parallel_hierarchy.NumAxes() == 1 && "
                 "reduced_out_parallel_hierarchy.NumAxes() == 1";
    const SbpParallel& in_sbp_parallel = in_parallel_distribution.sbp_parallel(0);
    const SbpParallel& out_sbp_parallel = out_parallel_distribution.sbp_parallel(0);
    Maybe<SubTskGphBuilderStatus> boxing_builder_status = TRY(sub_tsk_gph_builder_->Build(
        ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
        out_parallel_desc, lbi, logical_blob_desc, in_sbp_parallel, out_sbp_parallel, time_shape));
    LOG(INFO) << " builder_name: " << CHECK_JUST(boxing_builder_status)->builder_name() << "   "
              << CHECK_JUST(boxing_builder_status)->comment();
    return boxing_builder_status;
  } else if (reduced_in_parallel_hierarchy == reduced_out_parallel_hierarchy) {
    //(2, 3)[s0, s1]->(2, 3)[B, B]
    LOG(INFO) << "reduced_in_parallel_hierarchy == reduced_out_parallel_hierarchy";
    return BuildSameParallelHierarchySubTskGph(
        ctx, sub_tsk_gph_builder_, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
        in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc, reduced_in_parallel_hierarchy,
        reduced_out_parallel_hierarchy, reduced_in_parallel_distribution,
        reduced_out_parallel_distribution, time_shape);
  } else if (reduced_in_parallel_hierarchy.elem_cnt()
             == reduced_out_parallel_hierarchy.elem_cnt()) {
    //(2, 3)[s0, s1]->(3,2)[B, B]
    //(2, 3)[s0, s1]->(6)[s1] :  ->(2, 3)[s1, s1] <->(6)[s1]
    //(6)[s0]->(3,2)[s0, B]  (6)[s0]<->(3, 2)[s0, s0]->(3, 2)[s0, B]
    if ((reduced_in_parallel_hierarchy.NumAxes() == 1
         || reduced_in_parallel_distribution.sbp_parallel(0)
                == reduced_in_parallel_distribution.sbp_parallel(1))
        && reduced_out_parallel_hierarchy.NumAxes() == 2) {
      //(6)[s0]<->(3, 2)[s0, s0]
      //(2, 3)[s0, s0]->(3,2)[s1, s0] (2, 3)[s0, s0]<->(3, 2)[s0,s0]->(3,2)[s1, s0]
      LOG(INFO) << "eg: (6)[s0]<->(3, 2)[s0, s0] , (2, 3)[s0, s0]->(3,2)[s1, s0]";
      Shape intermediate_in_parallel_hierarchy(reduced_out_parallel_hierarchy);
      ParallelDistribution intermediate_parallel_distribution;
      *intermediate_parallel_distribution.add_sbp_parallel() =
          reduced_in_parallel_distribution.sbp_parallel(0);
      *intermediate_parallel_distribution.add_sbp_parallel() =
          reduced_in_parallel_distribution.sbp_parallel(0);
      return BuildSameParallelHierarchySubTskGph(
          ctx, sub_tsk_gph_builder_, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
          in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc,
          intermediate_in_parallel_hierarchy, reduced_out_parallel_hierarchy,
          intermediate_parallel_distribution, reduced_out_parallel_distribution, time_shape);
    } else if (reduced_in_parallel_hierarchy.NumAxes() == 2
               && (reduced_out_parallel_hierarchy.NumAxes() == 1
                   || reduced_out_parallel_distribution.sbp_parallel(0)
                          == reduced_out_parallel_distribution.sbp_parallel(1))) {
      //(2, 3)[s0, s1]->(6)[s1] :  ->(2, 3)[s1, s1] <->(6)[s1]
      //(2, 3)[s0, s1]->(3, 2)[s1, s1] :  ->(2, 3)[s1, s1] <->(3, 2)[s1, s1]
      LOG(INFO) << "eg: (2, 3)[s0, s1]->(6)[s1] , (2, 3)[s0, s1]->(3, 2)[s1, s1]";
      Shape intermediate_out_parallel_hierarchy(reduced_in_parallel_hierarchy);
      ParallelDistribution intermediate_parallel_distribution;
      *intermediate_parallel_distribution.add_sbp_parallel() =
          reduced_out_parallel_distribution.sbp_parallel(0);
      *intermediate_parallel_distribution.add_sbp_parallel() =
          reduced_out_parallel_distribution.sbp_parallel(0);
      return BuildSameParallelHierarchySubTskGph(
          ctx, sub_tsk_gph_builder_, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
          in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc,
          reduced_in_parallel_hierarchy, intermediate_out_parallel_hierarchy,
          reduced_in_parallel_distribution, intermediate_parallel_distribution, time_shape);
    } else {
      //(2, 3)[s0, s1]->(3,2)[s1, s0]  (2, 3)[s0, s1]->(2,3)[s0,s0]<->(3,2)[s0,s0]->(3,2)[s0,s1]
      LOG(INFO) << "eg: (2, 3)[s0, s1]->(3,2)[s1, s0] ";
      //(2, 3)[s0, s1]->(2,3)[s0,s0]
      Shape intermediate_out_parallel_hierarchy(reduced_in_parallel_hierarchy);
      ParallelDistribution intermediate_parallel_distribution;
      SbpParallel s0_parallel;
      s0_parallel.mutable_split_parallel()->set_axis(0);
      *intermediate_parallel_distribution.add_sbp_parallel() = s0_parallel;
      *intermediate_parallel_distribution.add_sbp_parallel() = s0_parallel;
      std::vector<SubTskGphBuilderStatus> status;
      std::vector<TaskNode*> out_tasks;
      std::vector<std::vector<TaskNode*>> ctrl_tasks;
      Maybe<SubTskGphBuilderStatus> first_status = BuildSameParallelHierarchySubTskGph(
          ctx, sub_tsk_gph_builder_, sorted_in_tasks, &out_tasks, &ctrl_tasks, in_parallel_desc,
          out_parallel_desc, lbi, logical_blob_desc, reduced_in_parallel_hierarchy,
          intermediate_out_parallel_hierarchy, reduced_in_parallel_distribution,
          intermediate_parallel_distribution, time_shape);
      status.push_back(*CHECK_JUST(first_status));
      // todo: process ctrl nodes
      //(3, 2)[s0,s0]->(3,2)[s0,s1]
      Shape intermediate_in_parallel_hierarchy(reduced_out_parallel_hierarchy);
      Maybe<SubTskGphBuilderStatus> second_status = BuildSameParallelHierarchySubTskGph(
          ctx, sub_tsk_gph_builder_, out_tasks, sorted_out_tasks, sorted_ctrl_tasks,
          in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc,
          intermediate_in_parallel_hierarchy, reduced_out_parallel_hierarchy,
          intermediate_parallel_distribution, reduced_out_parallel_distribution, time_shape);
      status.push_back(*CHECK_JUST(second_status));
      Maybe<SubTskGphBuilderStatus> composed_status = MakeComposedSubTskGphBuilderStatus(status);
      return composed_status;
    }
  } else {
    //(2, 3)->(4, 5)
    UNIMPLEMENTED();
  }
  return Error::BoxingNotSupportedError();
}

}  // namespace oneflow
