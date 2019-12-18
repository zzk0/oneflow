#ifndef ONEFLOW_CORE_JOB_COMPLETER_JOB_COMPLETER_H_
#define ONEFLOW_CORE_JOB_COMPLETER_JOB_COMPLETER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

class JobCompleteCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobCompleteCtx)
  JobCompleteCtx(Job* job) : job_(job), is_dirty_(false) {
    op_graph_.reset(new OpGraph(*job_));
    job_builder_.reset(new JobBuilder(job_));
  }
  ~JobCompleteCtx() = default;

  Job* GetJob() {
    is_dirty_ = true;
    return job_;
  }

  std::shared_ptr<JobBuilder> GetJobBuilder() {
    is_dirty_ = true;
    return job_builder_;
  }

  std::shared_ptr<OpGraph> GetOpGraph() { return op_graph_; }

  void RefreshOpGraphIfNeeded() {
    if (is_dirty_) {
      op_graph_.reset(new OpGraph(*job_));
      job_builder_.reset(new JobBuilder(job_));
    }
    is_dirty_ = false;
  }

 private:
  Job* job_;
  std::shared_ptr<OpGraph> op_graph_;
  std::shared_ptr<JobBuilder> job_builder_;
  bool is_dirty_;
};

class JobCompleter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobCompleter);
  JobCompleter() = default;
  ~JobCompleter() = default;

  void Complete(Job* job) const;
};

class GenerateFacadeImplOpConfWrapperStruct final {
 public:
  using Func = std::function<void(const OpNode&, JobBuilder*)>;
  GenerateFacadeImplOpConfWrapperStruct(const Func& f) : func_(std::make_unique<Func>(f)) {}
  void Call(const OpNode& op_node, JobBuilder* job_builder) const {
    (*func_)(op_node, job_builder);
  }

 private:
  const std::unique_ptr<const Func> func_;
};

#define REGISTER_FACADE_IMPL(op_type_case, gen_grad_func)                                   \
  REGISTER_CLASS_CREATOR(op_type_case, GenerateFacadeImplOpConfWrapperStruct, ([] {         \
                           return new GenerateFacadeImplOpConfWrapperStruct(gen_grad_func); \
                         }))

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_JOB_COMPLETER_H_
