#ifndef ONEFLOW_CORE_JOB_EVENT_REPORT_UTIL_H_
#define ONEFLOW_CORE_JOB_EVENT_REPORT_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/actor/act_event.pb.h"
#include "oneflow/core/actor/msg_event.pb.h"
#include "oneflow/core/actor/kernel_event.pb.h"
#include "oneflow/core/job/id_manager.h"

namespace oneflow {

template<typename EventType>
void LoadEvents(const std::string& event_filepath,
                std::list<EventType>* events);
std::string Time2String(const double d);
std::string Time2HumanReadable(const double d);
std::string GetActorInfo(const Plan& plan, const int64_t& actor_id);
void GetMachineTimeDiff(const std::string& time_diff_filepath,
                        std::vector<double>& machine_time_diffs);
double GetDiffTime(const int64_t actor_id,
                   const std::vector<double> machine_time_diffs);
void ActorId2RegstId(
    const std::string& plan_filepath,
    HashMap<int64_t, std::vector<int64_t>>& actor_id2produced_regsts,
    HashMap<int64_t, std::vector<int64_t>>& actor_id2consumed_regsts);
struct RegstEvent {
  int64_t regst_desc_id;
  int64_t producer_id;
  int64_t consumer_id;
  int64_t piece_id;
  int64_t act_id;
  double to_consumer_time;
  double from_producer_time;
  double to_producer_time;
  double from_consumer_time;
};

void Msg2RegstEvents(const std::string& msg_event_filepath,
                     HashMap<std::string, RegstEvent>& regst_events,
                     const std::string& time_diff_filepath);
}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_EVENT_REPORT_UTIL_H_
