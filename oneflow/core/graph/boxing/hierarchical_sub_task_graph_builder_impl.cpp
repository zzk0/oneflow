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

#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder_impl.h"
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
#include "oneflow/core/common/id_util.h"
#include "oneflow/core/graph/id_serialization.h"
#include "oneflow/core/device/cpu_stream_index.h"
#ifdef WITH_CUDA
#include "oneflow/core/device/cuda_stream_index.h"
#endif

namespace oneflow {

namespace {

bool ParallelDistributionAllSameSplitParallel(const ParallelDistribution& parallel_distribution) {
  CHECK_GT(parallel_distribution.sbp_parallel_size(), 1);
  const SbpParallel& first_sbp = parallel_distribution.sbp_parallel(0);
  if (!first_sbp.has_split_parallel()) { return false; };
  FOR_RANGE(int64_t, i, 1, parallel_distribution.sbp_parallel_size()) {
    if (parallel_distribution.sbp_parallel(i) != first_sbp) { return false; }
  }
  return true;
}

void ParallelDimReduce(const ParallelDesc& parallel_desc,
                       const ParallelDistribution& parallel_distribution,
                       ParallelDesc* reduced_parallel_desc,
                       ParallelDistribution* reduced_parallel_distribution) {
  const auto& hierarchy = parallel_desc.hierarchy();
  DimVector reduced_hierarchy;
  reduced_hierarchy.push_back(hierarchy->At(0));
  *reduced_parallel_distribution->add_sbp_parallel() = parallel_distribution.sbp_parallel(0);
  FOR_RANGE(int64_t, i, 1, hierarchy->NumAxes()) {
    if (parallel_distribution.sbp_parallel(i) == parallel_distribution.sbp_parallel(i - 1)) {
      reduced_hierarchy.back() *= hierarchy->At(i);
    } else {
      reduced_hierarchy.push_back(hierarchy->At(i));
      *reduced_parallel_distribution->add_sbp_parallel() = parallel_distribution.sbp_parallel(i);
    }
  }
  ParallelConf reduced_parallel_conf = parallel_desc.parallel_conf();
  Shape(reduced_hierarchy).ToProto(reduced_parallel_conf.mutable_hierarchy());
  *reduced_parallel_desc = ParallelDesc(reduced_parallel_conf);
}

void CollaborativeParallelDimReduce(const ParallelDesc& in_parallel_desc,
                                    const ParallelDesc& out_parallel_desc,
                                    const ParallelDistribution& in_parallel_distribution,
                                    const ParallelDistribution& out_parallel_distribution,
                                    ParallelDesc* reduced_in_parallel_desc,
                                    ParallelDesc* reduced_out_parallel_desc,
                                    ParallelDistribution* reduced_in_parallel_distribution,
                                    ParallelDistribution* reduced_out_parallel_distribution) {
  const auto& in_hierarchy = in_parallel_desc.hierarchy();
  const auto& out_hierarchy = out_parallel_desc.hierarchy();
  CHECK_EQ(in_hierarchy->NumAxes(), out_hierarchy->NumAxes());

  DimVector reduced_in_hierarchy;
  reduced_in_hierarchy.push_back(in_hierarchy->At(0));
  *reduced_in_parallel_distribution->add_sbp_parallel() = in_parallel_distribution.sbp_parallel(0);

  DimVector reduced_out_hierarchy;
  reduced_out_hierarchy.push_back(out_hierarchy->At(0));
  *reduced_out_parallel_distribution->add_sbp_parallel() =
      out_parallel_distribution.sbp_parallel(0);

  FOR_RANGE(int64_t, i, 1, in_hierarchy->NumAxes()) {
    if ((in_parallel_distribution.sbp_parallel(i) == in_parallel_distribution.sbp_parallel(i - 1))
        && (out_parallel_distribution.sbp_parallel(i)
            == out_parallel_distribution.sbp_parallel(i - 1))) {
      reduced_in_hierarchy.back() *= in_hierarchy->At(i);
      reduced_out_hierarchy.back() *= out_hierarchy->At(i);
    } else {
      reduced_in_hierarchy.push_back(in_hierarchy->At(i));
      *reduced_in_parallel_distribution->add_sbp_parallel() =
          in_parallel_distribution.sbp_parallel(i);

      reduced_out_hierarchy.push_back(out_hierarchy->At(i));
      *reduced_out_parallel_distribution->add_sbp_parallel() =
          out_parallel_distribution.sbp_parallel(i);
    }
  }

  ParallelConf reduced_in_parallel_conf = in_parallel_desc.parallel_conf();
  Shape(reduced_in_hierarchy).ToProto(reduced_in_parallel_conf.mutable_hierarchy());
  *reduced_in_parallel_desc = ParallelDesc(reduced_in_parallel_conf);

  ParallelConf reduced_out_parallel_conf = out_parallel_desc.parallel_conf();
  Shape(reduced_out_hierarchy).ToProto(reduced_out_parallel_conf.mutable_hierarchy());
  *reduced_out_parallel_desc = ParallelDesc(reduced_out_parallel_conf);
}

void InOutParallelDimReduce(const ParallelDesc& in_parallel_desc,
                            const ParallelDesc& out_parallel_desc,
                            const ParallelDistribution& in_parallel_distribution,
                            const ParallelDistribution& out_parallel_distribution,
                            ParallelDesc* reduced_in_parallel_desc,
                            ParallelDesc* reduced_out_parallel_desc,
                            ParallelDistribution* reduced_in_parallel_distribution,
                            ParallelDistribution* reduced_out_parallel_distribution) {
  const int64_t in_hierarchy_axes = in_parallel_desc.hierarchy()->NumAxes();
  const int64_t out_hierarchy_axes = out_parallel_desc.hierarchy()->NumAxes();
  if (in_hierarchy_axes == 1 && out_hierarchy_axes == 1) {
    *reduced_in_parallel_desc = in_parallel_desc;
    *reduced_out_parallel_desc = out_parallel_desc;
    *reduced_in_parallel_distribution = in_parallel_distribution;
    *reduced_out_parallel_distribution = out_parallel_distribution;
  } else if (in_hierarchy_axes != out_hierarchy_axes) {
    ParallelDimReduce(in_parallel_desc, in_parallel_distribution, reduced_in_parallel_desc,
                      reduced_in_parallel_distribution);
    ParallelDimReduce(out_parallel_desc, out_parallel_distribution, reduced_out_parallel_desc,
                      reduced_out_parallel_distribution);
  } else {
    CollaborativeParallelDimReduce(in_parallel_desc, out_parallel_desc, in_parallel_distribution,
                                   out_parallel_distribution, reduced_in_parallel_desc,
                                   reduced_out_parallel_desc, reduced_in_parallel_distribution,
                                   reduced_out_parallel_distribution);
  }
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

  const auto GetBoxingGpuThrdId = [](int64_t machine_id, int64_t dev_id,
                                     CudaWorkType work_type) -> int64_t {
    int64_t thrd_id = -1;
#ifdef WITH_CUDA
    DeviceId device_id{static_cast<DeviceId::rank_t>(machine_id), DeviceType::kGPU,
                       static_cast<DeviceId::device_index_t>(dev_id)};
    auto* generator = dynamic_cast<CudaStreamIndexGenerator*>(
        Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetGenerator(device_id));
    CHECK_NOTNULL(generator);
    StreamId::stream_index_t stream_index = 0;
    if (work_type == CudaWorkType::kCopyH2D) {
      stream_index = generator->GenerateH2DStreamIndex();
    } else if (work_type == CudaWorkType::kCopyD2H) {
      stream_index = generator->GenerateD2HStreamIndex();
    } else if (work_type == CudaWorkType::kMix) {
      stream_index = generator->GenerateMixStreamIndex();
    } else {
      UNIMPLEMENTED();
    }
    thrd_id = SerializeStreamIdToInt64(StreamId{device_id, stream_index});
#else
    UNIMPLEMENTED();
#endif
    return thrd_id;
  };
  const auto NewEdge = [&ctx]() -> TaskEdge* { return ctx->task_graph()->NewEdge(); };
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
      int64_t dev_id = CHECK_JUST(pd.DeviceId4ParallelId(parallel_id));
      thrd_id = GetBoxingGpuThrdId(machine_id, dev_id, CudaWorkType::kCopyH2D);
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
  const std::vector<TensorSliceView> in_slices = SubTskGphBuilderUtil::GetTensorSliceView(
      in_parallel_hierarchy, in_parallel_distribution, logical_blob_desc.shape());
  const std::vector<TensorSliceView> out_slices = SubTskGphBuilderUtil::GetTensorSliceView(
      out_parallel_hierarchy, out_parallel_distribution, logical_blob_desc.shape());
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

Maybe<SubTskGphBuilderStatus> BuildSliceBoxingAddSubTskGph(
    SubTskGphBuilderCtx* ctx, const std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder,
    const std::vector<TaskNode*>& sorted_in_tasks, std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const Shape& in_parallel_hierarchy,
    const Shape& out_parallel_hierarchy, const ParallelDistribution& in_parallel_distribution,
    const ParallelDistribution& out_parallel_distribution, const Shape& time_shape) {
  CHECK_OR_RETURN(in_parallel_distribution.sbp_parallel(0).has_partial_sum_parallel());
  const auto GetBoxingGpuThrdId = [](int64_t machine_id, int64_t dev_id,
                                     CudaWorkType work_type) -> int64_t {
    int64_t thrd_id = -1;
#ifdef WITH_CUDA
    DeviceId device_id{static_cast<DeviceId::rank_t>(machine_id), DeviceType::kGPU,
                       static_cast<DeviceId::device_index_t>(dev_id)};
    auto* generator = dynamic_cast<CudaStreamIndexGenerator*>(
        Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetGenerator(device_id));
    CHECK_NOTNULL(generator);
    StreamId::stream_index_t stream_index = 0;
    if (work_type == CudaWorkType::kCopyH2D) {
      stream_index = generator->GenerateH2DStreamIndex();
    } else if (work_type == CudaWorkType::kCopyD2H) {
      stream_index = generator->GenerateD2HStreamIndex();
    } else if (work_type == CudaWorkType::kMix) {
      stream_index = generator->GenerateMixStreamIndex();
    } else {
      UNIMPLEMENTED();
    }
    thrd_id = SerializeStreamIdToInt64(StreamId{device_id, stream_index});
#else
    UNIMPLEMENTED();
#endif
    return thrd_id;
  };
  const auto NewEdge = [&ctx]() -> TaskEdge* { return ctx->task_graph()->NewEdge(); };
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
      int64_t dev_id = CHECK_JUST(pd.DeviceId4ParallelId(parallel_id));
      thrd_id = GetBoxingGpuThrdId(machine_id, dev_id, CudaWorkType::kCopyH2D);
#else
      UNIMPLEMENTED();
#endif
    } else {
      UNIMPLEMENTED();
    }
    node->Init(lbi, slice, mode, machine_id, thrd_id);
    return node;
  };
  const auto FindFirstNotEmptyInterSection =
      [](const TensorSliceView& out_slice,
         const std::vector<TensorSliceView>& in_slices) -> TensorSliceView {
    FOR_RANGE(int64_t, in_id, 0, in_slices.size()) {
      const TensorSliceView& intersection = out_slice.Intersect(in_slices.at(in_id));
      if (intersection.IsEmpty()) { continue; }
      return intersection;
    }
  };

  LOG(INFO) << "Build2DSliceBoxingAddSubTskGph";
  // can not process P, B->B, B, only axis 1 has split
  const std::vector<TensorSliceView> in_slices = SubTskGphBuilderUtil::GetTensorSliceView(
      in_parallel_hierarchy, in_parallel_distribution, logical_blob_desc.shape());
  const std::vector<TensorSliceView> out_slices = SubTskGphBuilderUtil::GetTensorSliceView(
      out_parallel_hierarchy, out_parallel_distribution, logical_blob_desc.shape());
  const int64_t in_parallel_num = in_parallel_desc.parallel_num();
  const int64_t out_parallel_num = out_parallel_desc.parallel_num();
  FOR_RANGE(int64_t, out_id, 0, out_parallel_num) {
    const TensorSliceView& out_slice = out_slices.at(out_id);
    SliceBoxingTaskNode* out_node =
        CreateBoxingNode121(out_parallel_desc, out_id, out_slice, kSliceBoxingTaskModeCopy);
    const TensorSliceView& first_intersection = FindFirstNotEmptyInterSection(out_slice, in_slices);
    SliceBoxingTaskNode* add_node =
        CreateBoxingNode121(out_parallel_desc, out_id, first_intersection, kSliceBoxingTaskModeAdd);
    FOR_RANGE(int64_t, in_id, 0, in_parallel_num) {
      const TensorSliceView& in_slice = in_slices.at(in_id);
      const TensorSliceView& intersection = out_slice.Intersect(in_slice);
      if (intersection.IsEmpty()) { continue; }
      CHECK_OR_RETURN(intersection == first_intersection);
      TaskNode* in_node = sorted_in_tasks.at(in_id);
      SliceBoxingTaskNode* in_copy_node =
          CreateBoxingNode121(in_parallel_desc, in_id, intersection, kSliceBoxingTaskModeCopy);
      in_copy_node->ConnectToSrcNodeWithSlice(in_node, NewEdge(), in_slice);
      TaskNode* proxy_node = ctx->GetProxyNode(in_copy_node, in_copy_node->MemZoneId121(),
                                               out_node->machine_id(), out_node->MemZoneId121());
      add_node->ConnectToSrcNodeWithSlice(proxy_node, NewEdge(), intersection);
    }
    out_node->ConnectToSrcNodeWithSlice(add_node, NewEdge(), first_intersection);
    sorted_out_tasks->push_back(out_node);
  }
  return TRY(BuildSubTskGphBuilderStatus("2DSliceBoxingAddSubTskGphBuilder", "P2S"));
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
  if ((in_parallel_distribution.sbp_parallel(0) == in_parallel_distribution.sbp_parallel(1)
       && in_parallel_distribution.sbp_parallel(0).has_split_parallel())
      || (out_parallel_distribution.sbp_parallel(0) == out_parallel_distribution.sbp_parallel(1)
          && out_parallel_distribution.sbp_parallel(0).has_split_parallel())) {
    if (in_parallel_distribution.sbp_parallel(0).has_partial_sum_parallel()) {
      LOG(INFO) << " slice boxing add";
      return BuildSliceBoxingAddSubTskGph(
          ctx, sub_tsk_gph_builder, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
          in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc, in_parallel_hierarchy,
          out_parallel_hierarchy, in_parallel_distribution, out_parallel_distribution, time_shape);
    } else {
      LOG(INFO) << " slice boxing copy";
      return Build2DSliceBoxingSubTskGph(
          ctx, sub_tsk_gph_builder, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
          in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc, in_parallel_hierarchy,
          out_parallel_hierarchy, in_parallel_distribution, out_parallel_distribution, time_shape);
    }
  } else {
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
      Maybe<SubTskGphBuilderStatus> boxing_builder_status = TRY(sub_tsk_gph_builder->Build(
          ctx, in_tasks, &out_tasks, &ctrl_tasks, sub_builder_in_parallel_desc,
          sub_builder_out_parallel_desc, lbi, new_blob_desc, in_sbp_parallel, out_sbp_parallel,
          time_shape));
      LOG(INFO) << " builder_name: " << CHECK_JUST(boxing_builder_status)->builder_name() << "   "
                << CHECK_JUST(boxing_builder_status)->comment();

      status.push_back(*CHECK_JUST(boxing_builder_status));

      sorted_out_tasks->resize(out_parallel_desc.parallel_num());
      CHECK_EQ_OR_RETURN(out_tasks.size(), in_parallel_hierarchy.At(0));
      FOR_RANGE(int64_t, j, 0, in_parallel_hierarchy.At(0)) {
        const int64_t parallel_id = j * out_parallel_hierarchy.At(1) + i;
        sorted_out_tasks->at(parallel_id) = out_tasks.at(j);
        for (TaskNode* ctrl_node : ctrl_tasks.at(j)) {
          sorted_ctrl_tasks->at(parallel_id).push_back(ctrl_node);
        }
      }
    }
    Maybe<SubTskGphBuilderStatus> composed_status = MakeComposedSubTskGphBuilderStatus(status);
    return composed_status;
  }
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

  LOG(INFO) << "2D sbp axis 1 BuildLastAxisParallelDistributionChangeSubGph";
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

Maybe<SubTskGphBuilderStatus> Build1dParallelHierarchySubTskGph(
    SubTskGphBuilderCtx* ctx, const std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder,
    const std::vector<TaskNode*>& sorted_in_tasks, std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const Shape& in_parallel_hierarchy,
    const Shape& out_parallel_hierarchy, const ParallelDistribution& in_parallel_distribution,
    const ParallelDistribution& out_parallel_distribution, const Shape& time_shape) {
  Maybe<SubTskGphBuilderStatus> boxing_builder_status = TRY(sub_tsk_gph_builder->Build(
      ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
      out_parallel_desc, lbi, logical_blob_desc, in_parallel_distribution.sbp_parallel(0),
      out_parallel_distribution.sbp_parallel(0), time_shape));
  return boxing_builder_status;
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
  if (in_parallel_distribution.sbp_parallel(0) == out_parallel_distribution.sbp_parallel(0)) {
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
  } else if (in_parallel_distribution.sbp_parallel(0).has_split_parallel()
             && in_parallel_distribution.sbp_parallel(1).has_split_parallel()
             && out_parallel_distribution.sbp_parallel(0).has_split_parallel()
             && out_parallel_distribution.sbp_parallel(1).has_split_parallel()) {
    return Build2DSliceBoxingSubTskGph(
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
        ctx, sub_tsk_gph_builder, out_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
        out_parallel_desc, lbi, logical_blob_desc, in_parallel_hierarchy, out_parallel_hierarchy,
        intermediate_parallel_distribution, out_parallel_distribution, time_shape);
    status.push_back(*CHECK_JUST(second_status));
    Maybe<SubTskGphBuilderStatus> composed_status = MakeComposedSubTskGphBuilderStatus(status);
    return composed_status;
  }
}

Maybe<SubTskGphBuilderStatus> BuildSameElemcntParallelHierarchySubTskGph(
    SubTskGphBuilderCtx* ctx, const std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder,
    const std::vector<TaskNode*>& sorted_in_tasks, std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const Shape& in_parallel_hierarchy,
    const Shape& out_parallel_hierarchy, const ParallelDistribution& in_parallel_distribution,
    const ParallelDistribution& out_parallel_distribution, const Shape& time_shape) {
  if ((in_parallel_hierarchy.NumAxes() == 1
       || in_parallel_distribution.sbp_parallel(0) == in_parallel_distribution.sbp_parallel(1))
      && out_parallel_hierarchy.NumAxes() == 2) {
    //(6)[s0]<->(3, 2)[s0, s0]
    //(2, 3)[s0, s0]->(3,2)[s1, s0] (2, 3)[s0, s0]<->(3, 2)[s0,s0]->(3,2)[s1, s0]
    LOG(INFO) << "eg: (6)[s0]<->(3, 2)[s0, s0] , (2, 3)[s0, s0]->(3,2)[s1, s0]";
    Shape intermediate_in_parallel_hierarchy(out_parallel_hierarchy);
    ParallelDistribution intermediate_parallel_distribution;
    *intermediate_parallel_distribution.add_sbp_parallel() =
        in_parallel_distribution.sbp_parallel(0);
    *intermediate_parallel_distribution.add_sbp_parallel() =
        in_parallel_distribution.sbp_parallel(0);
    return BuildSameParallelHierarchySubTskGph(
        ctx, sub_tsk_gph_builder, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
        in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc,
        intermediate_in_parallel_hierarchy, out_parallel_hierarchy,
        intermediate_parallel_distribution, out_parallel_distribution, time_shape);
  } else if (in_parallel_hierarchy.NumAxes() == 2
             && (out_parallel_hierarchy.NumAxes() == 1
                 || out_parallel_distribution.sbp_parallel(0)
                        == out_parallel_distribution.sbp_parallel(1))) {
    //(2, 3)[s0, s1]->(6)[s1] :  ->(2, 3)[s1, s1] <->(6)[s1]
    //(2, 3)[s0, s1]->(3, 2)[s1, s1] :  ->(2, 3)[s1, s1] <->(3, 2)[s1, s1]
    LOG(INFO) << "eg: (2, 3)[s0, s1]->(6)[s1] , (2, 3)[s0, s1]->(3, 2)[s1, s1]";
    Shape intermediate_out_parallel_hierarchy(in_parallel_hierarchy);
    ParallelDistribution intermediate_parallel_distribution;
    *intermediate_parallel_distribution.add_sbp_parallel() =
        out_parallel_distribution.sbp_parallel(0);
    *intermediate_parallel_distribution.add_sbp_parallel() =
        out_parallel_distribution.sbp_parallel(0);
    return BuildSameParallelHierarchySubTskGph(
        ctx, sub_tsk_gph_builder, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
        in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc, in_parallel_hierarchy,
        intermediate_out_parallel_hierarchy, in_parallel_distribution,
        intermediate_parallel_distribution, time_shape);
  } else {
    //(2, 3)[s0, s1]->(3,2)[s1, s0]  (2, 3)[s0, s1]->(2,3)[s0,s0]<->(3,2)[s0,s0]->(3,2)[s0,s1]
    LOG(INFO) << "eg: (2, 3)[s0, s1]->(3,2)[s1, s0] ";
    //(2, 3)[s0, s1]->(2,3)[s0,s0]
    Shape intermediate_out_parallel_hierarchy(in_parallel_hierarchy);
    ParallelDistribution intermediate_parallel_distribution;
    SbpParallel s0_parallel;
    s0_parallel.mutable_split_parallel()->set_axis(0);
    *intermediate_parallel_distribution.add_sbp_parallel() = s0_parallel;
    *intermediate_parallel_distribution.add_sbp_parallel() = s0_parallel;
    std::vector<SubTskGphBuilderStatus> status;
    std::vector<TaskNode*> out_tasks;
    std::vector<std::vector<TaskNode*>> ctrl_tasks;
    Maybe<SubTskGphBuilderStatus> first_status = BuildSameParallelHierarchySubTskGph(
        ctx, sub_tsk_gph_builder, sorted_in_tasks, &out_tasks, &ctrl_tasks, in_parallel_desc,
        out_parallel_desc, lbi, logical_blob_desc, in_parallel_hierarchy,
        intermediate_out_parallel_hierarchy, in_parallel_distribution,
        intermediate_parallel_distribution, time_shape);
    status.push_back(*CHECK_JUST(first_status));
    // todo: process ctrl nodes
    //(3, 2)[s0,s0]->(3,2)[s0,s1]
    Shape intermediate_in_parallel_hierarchy(out_parallel_hierarchy);
    Maybe<SubTskGphBuilderStatus> second_status = BuildSameParallelHierarchySubTskGph(
        ctx, sub_tsk_gph_builder, out_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
        out_parallel_desc, lbi, logical_blob_desc, intermediate_in_parallel_hierarchy,
        out_parallel_hierarchy, intermediate_parallel_distribution, out_parallel_distribution,
        time_shape);
    status.push_back(*CHECK_JUST(second_status));
    Maybe<SubTskGphBuilderStatus> composed_status = MakeComposedSubTskGphBuilderStatus(status);
    return composed_status;
  }
}

std::shared_ptr<ChainSubTskGphBuilder> Build1DSubTskGphBuilder() {
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
  builders.emplace_back(new OneToOneSubTskGphBuilder());
  builders.emplace_back(new B21SubTskGphBuilder());
  if (!Global<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()) {
    builders.emplace_back(new CollectiveBoxingSubTskGphBuilder());
  }
  builders.emplace_back(new SliceBoxingSubTskGphBuilder());
  builders.emplace_back(new NaiveB2BSubTskGphBuilder());
  builders.emplace_back(new NaiveB2PSubTskGphBuilder());
  return std::make_shared<ChainSubTskGphBuilder>(builders);
}

}  // namespace

class FlatSubTskGphBuilder final : public HierarchicalSubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FlatSubTskGphBuilder);
  FlatSubTskGphBuilder() { sub_tsk_gph_builder_ = Build1DSubTskGphBuilder(); }
  ~FlatSubTskGphBuilder() = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<TaskNode*>& sorted_in_tasks,
                                      std::vector<TaskNode*>* sorted_out_tasks,
                                      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks,
                                      const ParallelDesc& in_parallel_desc,
                                      const ParallelDesc& out_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const ParallelDistribution& in_parallel_distribution,
                                      const ParallelDistribution& out_parallel_distribution,
                                      const Shape& time_shape) const override {
    return JUST(sub_tsk_gph_builder_->Build(
        ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
        out_parallel_desc, lbi, logical_blob_desc, in_parallel_distribution.sbp_parallel(0),
        out_parallel_distribution.sbp_parallel(0), time_shape));
  }

 private:
  std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder_;
};

class IntraGroupSubTskGphBuilder final : public HierarchicalSubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IntraGroupSubTskGphBuilder);
  IntraGroupSubTskGphBuilder() { sub_tsk_gph_builder_ = Build1DSubTskGphBuilder(); }
  ~IntraGroupSubTskGphBuilder() = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<TaskNode*>& sorted_in_tasks,
                                      std::vector<TaskNode*>* sorted_out_tasks,
                                      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks,
                                      const ParallelDesc& in_parallel_desc,
                                      const ParallelDesc& out_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const ParallelDistribution& in_parallel_distribution,
                                      const ParallelDistribution& out_parallel_distribution,
                                      const Shape& time_shape) const override {
    CHECK_EQ(*in_parallel_desc.hierarchy(), *out_parallel_desc.hierarchy());
    const auto& hierarchy = in_parallel_desc.hierarchy();
    CHECK_EQ(hierarchy->NumAxes(), 2);
    std::vector<SubTskGphBuilderStatus> status;
    const int64_t num_groups = hierarchy->At(0);
    const int64_t group_size = hierarchy->At(1);
    sorted_ctrl_tasks->resize(out_parallel_desc.parallel_num());
    sorted_out_tasks->resize(out_parallel_desc.parallel_num());
    FOR_RANGE(int64_t, i, 0, num_groups) {
      LOG(ERROR) << "IntraGroupSubTskGphBuilder " << i;
      std::vector<TaskNode*> in_tasks;
      std::vector<TaskNode*> out_tasks;
      std::vector<std::vector<TaskNode*>> ctrl_tasks;
      ParallelConf in_parallel_conf;
      in_parallel_conf.set_device_tag(in_parallel_desc.device_tag());
      in_parallel_conf.mutable_hierarchy()->add_dim(group_size);
      ParallelConf out_parallel_conf;
      out_parallel_conf.set_device_tag(out_parallel_desc.device_tag());
      out_parallel_conf.mutable_hierarchy()->add_dim(group_size);
      FOR_RANGE(int64_t, j, 0, group_size) {
        const int64_t parallel_id = i * group_size + j;
        in_tasks.push_back(sorted_in_tasks.at(parallel_id));
        in_parallel_conf.add_device_name(
            std::to_string(CHECK_JUST(in_parallel_desc.MachineId4ParallelId(parallel_id))) + ":"
            + std::to_string(CHECK_JUST(in_parallel_desc.DeviceId4ParallelId(parallel_id))));
        out_parallel_conf.add_device_name(
            std::to_string(CHECK_JUST(out_parallel_desc.MachineId4ParallelId(parallel_id))) + ":"
            + std::to_string(CHECK_JUST(out_parallel_desc.DeviceId4ParallelId(parallel_id))));
      }
      DimVector dim_vec = logical_blob_desc.shape().dim_vec();
      if (in_parallel_distribution.sbp_parallel(0).has_split_parallel()) {
        const int64_t axis = in_parallel_distribution.sbp_parallel(0).split_parallel().axis();
        dim_vec.at(axis) /= hierarchy->At(0);
      }
      BlobDesc new_blob_desc(Shape(dim_vec), logical_blob_desc.data_type());
      Maybe<SubTskGphBuilderStatus> boxing_builder_status = JUST(sub_tsk_gph_builder_->Build(
          ctx, in_tasks, &out_tasks, &ctrl_tasks, ParallelDesc(in_parallel_conf),
          ParallelDesc(out_parallel_conf), lbi, new_blob_desc,
          in_parallel_distribution.sbp_parallel(1), out_parallel_distribution.sbp_parallel(1),
          time_shape));
      status.push_back(*CHECK_JUST(boxing_builder_status));
      CHECK_EQ_OR_RETURN(out_tasks.size(), group_size);
      FOR_RANGE(int64_t, j, 0, group_size) {
        const int64_t parallel_id = i * group_size + j;
        sorted_out_tasks->at(parallel_id) = out_tasks.at(j);
        if (!ctrl_tasks.empty()) {
          for (TaskNode* ctrl_node : ctrl_tasks.at(j)) {
            sorted_ctrl_tasks->at(parallel_id).push_back(ctrl_node);
          }
        }
      }
    }
    Maybe<SubTskGphBuilderStatus> composed_status = MakeComposedSubTskGphBuilderStatus(status);
    return composed_status;
  }

 private:
  std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder_;
};

class InterGroupSubTskGphBuilder final : public HierarchicalSubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InterGroupSubTskGphBuilder);
  InterGroupSubTskGphBuilder() { sub_tsk_gph_builder_ = Build1DSubTskGphBuilder(); }
  ~InterGroupSubTskGphBuilder() = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<TaskNode*>& sorted_in_tasks,
                                      std::vector<TaskNode*>* sorted_out_tasks,
                                      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks,
                                      const ParallelDesc& in_parallel_desc,
                                      const ParallelDesc& out_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const ParallelDistribution& in_parallel_distribution,
                                      const ParallelDistribution& out_parallel_distribution,
                                      const Shape& time_shape) const override {
    CHECK_EQ(*in_parallel_desc.hierarchy(), *out_parallel_desc.hierarchy());
    const auto& hierarchy = in_parallel_desc.hierarchy();
    CHECK_EQ(hierarchy->NumAxes(), 2);
    std::vector<SubTskGphBuilderStatus> status;
    const int64_t num_groups = hierarchy->At(0);
    const int64_t group_size = hierarchy->At(1);
    sorted_ctrl_tasks->resize(out_parallel_desc.parallel_num());
    sorted_out_tasks->resize(out_parallel_desc.parallel_num());
    FOR_RANGE(int64_t, i, 0, group_size) {
      LOG(ERROR) << "InterGroupSubTskGphBuilder " << i;
      std::vector<TaskNode*> in_tasks;
      std::vector<TaskNode*> out_tasks;
      std::vector<std::vector<TaskNode*>> ctrl_tasks;
      ParallelConf in_parallel_conf;
      in_parallel_conf.set_device_tag(in_parallel_desc.device_tag());
      in_parallel_conf.mutable_hierarchy()->add_dim(num_groups);
      ParallelConf out_parallel_conf;
      out_parallel_conf.set_device_tag(out_parallel_desc.device_tag());
      out_parallel_conf.mutable_hierarchy()->add_dim(num_groups);
      FOR_RANGE(int64_t, j, 0, num_groups) {
        const int64_t parallel_id = j * group_size + i;
        in_tasks.push_back(sorted_in_tasks.at(parallel_id));
        in_parallel_conf.add_device_name(
            std::to_string(CHECK_JUST(in_parallel_desc.MachineId4ParallelId(parallel_id))) + ":"
            + std::to_string(CHECK_JUST(in_parallel_desc.DeviceId4ParallelId(parallel_id))));
        out_parallel_conf.add_device_name(
            std::to_string(CHECK_JUST(out_parallel_desc.MachineId4ParallelId(parallel_id))) + ":"
            + std::to_string(CHECK_JUST(out_parallel_desc.DeviceId4ParallelId(parallel_id))));
      }
      DimVector dim_vec = logical_blob_desc.shape().dim_vec();
      if (in_parallel_distribution.sbp_parallel(1).has_split_parallel()) {
        const int64_t axis = in_parallel_distribution.sbp_parallel(1).split_parallel().axis();
        dim_vec.at(axis) /= hierarchy->At(1);
      }
      BlobDesc new_blob_desc(Shape(dim_vec), logical_blob_desc.data_type());
      Maybe<SubTskGphBuilderStatus> boxing_builder_status = JUST(sub_tsk_gph_builder_->Build(
          ctx, in_tasks, &out_tasks, &ctrl_tasks, ParallelDesc(in_parallel_conf),
          ParallelDesc(out_parallel_conf), lbi, new_blob_desc,
          in_parallel_distribution.sbp_parallel(0), out_parallel_distribution.sbp_parallel(0),
          time_shape));
      status.push_back(*CHECK_JUST(boxing_builder_status));

      CHECK_EQ_OR_RETURN(out_tasks.size(), num_groups);
      FOR_RANGE(int64_t, j, 0, num_groups) {
        const int64_t parallel_id = j * group_size + i;
        sorted_out_tasks->at(parallel_id) = out_tasks.at(j);
        if (!ctrl_tasks.empty()) {
          for (TaskNode* ctrl_node : ctrl_tasks.at(j)) {
            sorted_ctrl_tasks->at(parallel_id).push_back(ctrl_node);
          }
        }
      }
    }
    Maybe<SubTskGphBuilderStatus> composed_status = MakeComposedSubTskGphBuilderStatus(status);
    return composed_status;
  }

 private:
  std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder_;
};

struct DispatchHierarchicalSubTskGphBuilder::Impl {
  Impl();
  std::unique_ptr<FlatSubTskGphBuilder> flat_sub_tsk_gph_builder_;
  std::unique_ptr<IntraGroupSubTskGphBuilder> intra_group_sub_tsk_gph_builder_;
  std::unique_ptr<InterGroupSubTskGphBuilder> inter_group_sub_tsk_gph_builder_;
};

DispatchHierarchicalSubTskGphBuilder::Impl::Impl() {
  flat_sub_tsk_gph_builder_.reset(new FlatSubTskGphBuilder());
  intra_group_sub_tsk_gph_builder_.reset(new IntraGroupSubTskGphBuilder());
  inter_group_sub_tsk_gph_builder_.reset(new InterGroupSubTskGphBuilder());
}

DispatchHierarchicalSubTskGphBuilder::DispatchHierarchicalSubTskGphBuilder() {
  impl_.reset(new Impl());
}

DispatchHierarchicalSubTskGphBuilder::~DispatchHierarchicalSubTskGphBuilder() = default;

Maybe<SubTskGphBuilderStatus> DispatchHierarchicalSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const ParallelDistribution& in_parallel_distribution,
    const ParallelDistribution& out_parallel_distribution, const Shape& time_shape) const {
  ParallelDesc reduced_in_parallel_desc = in_parallel_desc;
  ParallelDesc reduced_out_parallel_desc = out_parallel_desc;
  ParallelDistribution reduced_in_parallel_distribution;
  ParallelDistribution reduced_out_parallel_distribution;
  InOutParallelDimReduce(in_parallel_desc, out_parallel_desc, in_parallel_distribution,
                         out_parallel_distribution, &reduced_in_parallel_desc,
                         &reduced_out_parallel_desc, &reduced_in_parallel_distribution,
                         &reduced_out_parallel_distribution);
  const auto& reduced_in_parallel_hierarchy = *reduced_in_parallel_desc.hierarchy();
  const auto& reduced_out_parallel_hierarchy = *reduced_out_parallel_desc.hierarchy();

  std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder_;
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
  builders.emplace_back(new OneToOneSubTskGphBuilder());
  builders.emplace_back(new B21SubTskGphBuilder());
  builders.emplace_back(new CollectiveBoxingSubTskGphBuilder());
  builders.emplace_back(new SliceBoxingSubTskGphBuilder());
  builders.emplace_back(new NaiveB2BSubTskGphBuilder());
  builders.emplace_back(new NaiveB2PSubTskGphBuilder());
  sub_tsk_gph_builder_.reset(new ChainSubTskGphBuilder(builders));

  if (reduced_in_parallel_hierarchy.NumAxes() == 1
      && reduced_out_parallel_hierarchy.NumAxes() == 1) {
    return impl_->flat_sub_tsk_gph_builder_->Build(
        ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, reduced_in_parallel_desc,
        reduced_out_parallel_desc, lbi, logical_blob_desc, reduced_in_parallel_distribution,
        reduced_out_parallel_distribution, time_shape);
  } else if (reduced_in_parallel_hierarchy == reduced_out_parallel_hierarchy) {
    if (reduced_in_parallel_distribution.sbp_parallel(0)
        == reduced_out_parallel_distribution.sbp_parallel(0)) {
      return impl_->intra_group_sub_tsk_gph_builder_->Build(
          ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, reduced_in_parallel_desc,
          reduced_out_parallel_desc, lbi, logical_blob_desc, reduced_in_parallel_distribution,
          reduced_out_parallel_distribution, time_shape);
    } else if (reduced_in_parallel_distribution.sbp_parallel(1)
                   == reduced_out_parallel_distribution.sbp_parallel(1)
               && !(ParallelDistributionAllSameSplitParallel(reduced_in_parallel_distribution)
                    || ParallelDistributionAllSameSplitParallel(
                        reduced_out_parallel_distribution))) {
      return impl_->inter_group_sub_tsk_gph_builder_->Build(
          ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, reduced_in_parallel_desc,
          reduced_out_parallel_desc, lbi, logical_blob_desc, reduced_in_parallel_distribution,
          reduced_out_parallel_distribution, time_shape);
    } else {
      //(2, 3)[s0, s1]->(2, 3)[B, B]
      LOG(INFO) << "reduced_in_parallel_hierarchy == reduced_out_parallel_hierarchy";
      return BuildSameParallelHierarchySubTskGph(
          ctx, sub_tsk_gph_builder_, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
          in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc,
          reduced_in_parallel_hierarchy, reduced_out_parallel_hierarchy,
          reduced_in_parallel_distribution, reduced_out_parallel_distribution, time_shape);
    }
  } else if (reduced_in_parallel_hierarchy.elem_cnt()
             == reduced_out_parallel_hierarchy.elem_cnt()) {
    //(2, 3)[s0, s1]->(3,2)[B, B]
    //(2, 3)[s0, s1]->(6)[s1] :  ->(2, 3)[s1, s1] <->(6)[s1]
    //(6)[s0]->(3,2)[s0, B]  (6)[s0]<->(3, 2)[s0, s0]->(3, 2)[s0, B]
    LOG(INFO)
        << "reduced_in_parallel_hierarchy.elem_cnt() == reduced_out_parallel_hierarchy.elem_cnt()";
    return BuildSameElemcntParallelHierarchySubTskGph(
        ctx, sub_tsk_gph_builder_, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
        in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc, reduced_in_parallel_hierarchy,
        reduced_out_parallel_hierarchy, reduced_in_parallel_distribution,
        reduced_out_parallel_distribution, time_shape);
  } else if (reduced_in_parallel_hierarchy.NumAxes() == 1) {
    LOG(INFO) << "reduced_in_parallel_hierarchy.elem_cnt() ！= "
                 "reduced_out_parallel_hierarchy.elem_cnt()， "
                 "reduced_in_parallel_hierarchy.NumAxes() == 1";
    LOG(INFO) << "eg: (1,)[B]->(2, 2)(S0, B):  (1)[B]->(4)[B]->(2,2)[S0, B]";
    Shape intermediate_parallel_hierarchy({reduced_out_parallel_hierarchy.elem_cnt()});
    ParallelDistribution intermediate_parallel_distribution;
    *intermediate_parallel_distribution.add_sbp_parallel() =
        reduced_in_parallel_distribution.sbp_parallel(0);
    ParallelDesc intermediate_parallel_desc = out_parallel_desc;

    std::vector<SubTskGphBuilderStatus> status;
    std::vector<TaskNode*> out_tasks;
    std::vector<std::vector<TaskNode*>> ctrl_tasks;
    //(1)[B]->(4)[B]
    Maybe<SubTskGphBuilderStatus> first_status = Build1dParallelHierarchySubTskGph(
        ctx, sub_tsk_gph_builder_, sorted_in_tasks, &out_tasks, &ctrl_tasks, in_parallel_desc,
        intermediate_parallel_desc, lbi, logical_blob_desc, reduced_in_parallel_hierarchy,
        intermediate_parallel_hierarchy, reduced_in_parallel_distribution,
        intermediate_parallel_distribution, time_shape);
    status.push_back(*CHECK_JUST(first_status));
    // todo: process ctrl nodes
    //(4)[B]->(2,2)[S0, B]
    Maybe<SubTskGphBuilderStatus> second_status = BuildSameElemcntParallelHierarchySubTskGph(
        ctx, sub_tsk_gph_builder_, out_tasks, sorted_out_tasks, sorted_ctrl_tasks,
        intermediate_parallel_desc, out_parallel_desc, lbi, logical_blob_desc,
        intermediate_parallel_hierarchy, reduced_out_parallel_hierarchy,
        intermediate_parallel_distribution, reduced_out_parallel_distribution, time_shape);
    status.push_back(*CHECK_JUST(second_status));
    Maybe<SubTskGphBuilderStatus> composed_status = MakeComposedSubTskGphBuilderStatus(status);
    return composed_status;
  } else {
    //(2, 3)->(4, 5)
    UNIMPLEMENTED();
  }
  return Error::BoxingNotSupportedError();
}

}  // namespace oneflow
