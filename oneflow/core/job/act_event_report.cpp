#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/actor/act_event.pb.h"

namespace oneflow {
// this function is exact same with ParseActEvents in improver.cpp
void LoadActEvents(const std::string& act_event_filepath,
                   std::list<ActEvent>* act_events) {
  NormalPersistentInStream in_stream(LocalFS(), act_event_filepath);
  size_t act_event_size;
  while (!in_stream.Read(reinterpret_cast<char*>(&act_event_size),
                         sizeof(size_t))) {
    std::vector<char> buffer(act_event_size);
    CHECK(!in_stream.Read(buffer.data(), act_event_size));
    act_events->emplace_back();
    act_events->back().ParseFromArray(buffer.data(), act_event_size);
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
void ActEventReport(const std::string& plan_filepath,
                    const std::string& act_event_filepath,
                    const std::string& report_filepath) {
  Global<JobDesc>::New();
  Plan plan;
  ParseProtoFromTextFile(plan_filepath, &plan);
  auto act_events = of_make_unique<std::list<ActEvent>>();
  LoadActEvents(act_event_filepath, act_events.get());
  // PersistentOutStream out_stream(LocalFS(), report_filepath);
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
