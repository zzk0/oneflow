#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/event_report_util.h"
namespace oneflow {

void ActEventAnalysis(const std::string& plan_filepath,
                      const std::string& act_event_filepath,
                      const std::string& msg_event_filepath,
                      const std::string& report_filepath) {
  Global<JobDesc>::New();
  Plan plan;
  ParseProtoFromTextFile(plan_filepath, &plan);
  HashMap<int64_t, int64_t> regst_id2produced_actor_id;
  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& regst_desc_it : task_proto.produced_regst_desc()) {
      regst_id2produced_actor_id.emplace(regst_desc_it.second.regst_desc_id(),
                                         task_proto.task_id());
    }
  }
  auto msg_events = of_make_unique<std::list<MsgEvent>>();
  LoadEvents<MsgEvent>(msg_event_filepath, msg_events.get());
  std::ofstream out_stream(report_filepath);
  out_stream
      << "src_actor,dst_actor,regst_desc_id,producer_id,piece_id,time,\n";
  for (auto event : *msg_events) {
    out_stream << "'" << std::to_string(event.src_actor_id()) + ",";
    out_stream << "'" << std::to_string(event.dst_actor_id()) + ",";
    out_stream << "'" << std::to_string(event.regst_desc_id()) + ",";
    out_stream << "'" << std::to_string(event.producer_actor_id()) + ",";
    out_stream << "'" << std::to_string(event.piece_id()) + ",";
    out_stream << "'" << Time2String(event.time()) + ",";
    out_stream << "\n";
  }
  out_stream.close();
  Global<JobDesc>::Delete();
}
}  // namespace oneflow

DEFINE_string(plan_filepath, "naive_plan", "");
DEFINE_string(msg_event_filepath, "msg_event.bin", "");
DEFINE_string(act_event_filepath, "act_event.bin", "");
DEFINE_string(report_filepath, "act_event_analysis.csv", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "make a msg report from " << FLAGS_msg_event_filepath;
  oneflow::ActEventAnalysis(FLAGS_plan_filepath, FLAGS_act_event_filepath,
                            FLAGS_msg_event_filepath, FLAGS_report_filepath);
  return 0;
}
