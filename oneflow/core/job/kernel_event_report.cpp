#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/event_report_util.h"

namespace oneflow {

void KernelEventReport(const std::string& plan_filepath,
                       const std::string& kernel_event_filepath,
                       const std::string& report_filepath) {
  Global<JobDesc>::New();
  Plan plan;
  ParseProtoFromTextFile(plan_filepath, &plan);
  auto kernel_events = of_make_unique<std::list<KernelEvent>>();
  LoadEvents<KernelEvent>(kernel_event_filepath, kernel_events.get());
  std::ofstream out_stream(report_filepath);
  out_stream << "actor_id,kernel,type,machine,thrd,act_id,start_time,stop_time,"
             << "run_time(s),run_time(ms)\n";
  for (auto event : *kernel_events) {
    out_stream << "'" << std::to_string(event.actor_id()) + ",";
    out_stream << event.name() + ",";
    out_stream << GetActorInfo(plan, event.actor_id()) + ",";
    out_stream << "'" << std::to_string(event.act_id()) + ",";
    out_stream << "'" << Time2String(event.start_time()) + ",";
    out_stream << "'" << Time2String(event.stop_time()) + ",";
    out_stream << Time2String(event.stop_time() - event.start_time()) + ",";
    out_stream << Time2HumanReadable(event.stop_time() - event.start_time())
                      + "\n";
  }
  out_stream.close();
  Global<JobDesc>::Delete();
}
}  // namespace oneflow

DEFINE_string(plan_filepath, "naive_plan", "");
DEFINE_string(kernel_event_filepath, "kernel_event.bin", "");
DEFINE_string(report_filepath, "kernel_event.csv", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "make a memory report from " << FLAGS_plan_filepath;
  oneflow::KernelEventReport(FLAGS_plan_filepath, FLAGS_kernel_event_filepath,
                             FLAGS_report_filepath);
  return 0;
}
