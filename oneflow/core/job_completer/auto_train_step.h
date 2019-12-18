#ifndef ONEFLOW_CORE_JOB_COMPLETER_AUTO_TRAIN_STEP_H_
#define ONEFLOW_CORE_JOB_COMPLETER_AUTO_TRAIN_STEP_H_

namespace oneflow {

class OpGraph;
class Job;
class JobCompleteCtx;

void AutoTrainStep(JobCompleteCtx* ctx);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_AUTO_TRAIN_STEP_H_
