#ifndef ONEFLOW_CORE_JOB_EVENT_REPORT_UTIL_H_
#define ONEFLOW_CORE_JOB_EVENT_REPORT_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/actor/act_event.pb.h"
#include "oneflow/core/actor/msg_event.pb.h"
#include "oneflow/core/actor/kernel_event.pb.h"

namespace oneflow {

template<typename EventType>
void LoadEvents(const std::string& event_filepath,
                std::list<EventType>* events);
std::string Time2String(const double d);
std::string Time2HumanReadable(const double d);
std::string GetActorInfo(const Plan& plan, const int64_t& actor_id);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_EVENT_REPORT_UTIL_H_
