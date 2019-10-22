#ifndef ONEFLOW_CORE_JOB_COMPLETER_SOFTMAX_LOSS_SPLIT_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_SOFTMAX_LOSS_SPLIT_PASS_H_

#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

class OpGraph;

class SoftmaxLossSplitPass final {
 public:
  SoftmaxLossSplitPass() = default;
  ~SoftmaxLossSplitPass() = default;
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_SOFTMAX_LOSS_SPLIT_PASS_H_
