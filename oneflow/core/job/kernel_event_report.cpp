#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/actor/kernel_event.pb.h"

namespace oneflow {
// this function is exact same with ParseKernelEvents in improver.cpp
void LoadKernelEvents(const std::string& kernel_event_filepath,
                      std::list<KernelEvent>* kernel_events) {
  NormalPersistentInStream in_stream(LocalFS(), kernel_event_filepath);
  size_t kernel_event_size;
  while (!in_stream.Read(reinterpret_cast<char*>(&kernel_event_size),
                         sizeof(size_t))) {
    std::vector<char> buffer(kernel_event_size);
    CHECK(!in_stream.Read(buffer.data(), kernel_event_size));
    kernel_events->emplace_back();
    kernel_events->back().ParseFromArray(buffer.data(), kernel_event_size);
  }
}

std::string Time2String(const double d) {
  std::stringstream stream;
  double t = d / 1e9;
  stream << std::fixed << std::setprecision(9) << t;
  return stream.str();  // s
}

std::string Time2HumanReadable(const double d) {
  std::stringstream stream;
  double t = d / 1e6;
  stream << std::fixed << std::setprecision(6) << t;
  return stream.str();  // ms
}

std::string GetActorInfo(const Plan& plan, const int64_t& actor_id) {
  for (const TaskProto& task : plan.task()) {
    if (task.task_id() == actor_id) {
      return TaskType_Name(task.task_type()) + ","
             + std::to_string(task.machine_id()) + ","
             + std::to_string(task.thrd_id());
    }
  }
  return ",,,";
}
void KernelEventReport(const std::string& plan_filepath,
                       const std::string& kernel_event_filepath,
                       const std::string& report_filepath) {
  Global<JobDesc>::New();
  Plan plan;
  ParseProtoFromTextFile(plan_filepath, &plan);
  auto kernel_events = of_make_unique<std::list<KernelEvent>>();
  LoadKernelEvents(kernel_event_filepath, kernel_events.get());
  // PersistentOutStream out_stream(LocalFS(), report_filepath);
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
