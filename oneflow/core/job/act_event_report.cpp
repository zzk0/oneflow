#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/event_report_util.h"

namespace oneflow {
void ActEventReport(const std::string& plan_filepath,
                    const std::string& act_event_filepath,
                    const std::string& report_filepath) {
  Global<JobDesc>::New();
  Plan plan;
  ParseProtoFromTextFile(plan_filepath, &plan);
  auto act_events = of_make_unique<std::list<ActEvent>>();
  LoadEvents<ActEvent>(act_event_filepath, act_events.get());
  std::ofstream out_stream(report_filepath);
  out_stream
      << "actor,type,machine,thrd,stream,act_id,push_time,start_time,stop_time,"
      << "block_time(s),block_time(ms),run_time(s),run_time(ms)\n";
  for (auto event : *act_events) {
    out_stream << "'" << std::to_string(event.actor_id()) + ",";
    out_stream << GetActorInfo(plan, event.actor_id()) + ",";
    out_stream << "'" << std::to_string(event.work_stream_id()) + ",";
    out_stream << "'" << std::to_string(event.act_id()) + ",";
    out_stream << "'" << Time2String(event.push_time()) + ",";
    out_stream << "'" << Time2String(event.start_time()) + ",";
    out_stream << "'" << Time2String(event.stop_time()) + ",";
    out_stream << Time2String(event.start_time() - event.push_time()) + ",";
    out_stream << Time2HumanReadable(event.start_time() - event.push_time())
                      + ",";
    out_stream << Time2String(event.stop_time() - event.start_time()) + ",";
    out_stream << Time2HumanReadable(event.stop_time() - event.start_time())
                      + "\n";
  }
  out_stream.close();
  Global<JobDesc>::Delete();
}
}  // namespace oneflow

DEFINE_string(plan_filepath, "naive_plan", "");
DEFINE_string(act_event_filepath, "act_event.bin", "");
DEFINE_string(report_filepath, "act_event.csv", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "make a memory report from " << FLAGS_plan_filepath;
  oneflow::ActEventReport(FLAGS_plan_filepath, FLAGS_act_event_filepath,
                          FLAGS_report_filepath);
  return 0;
}
