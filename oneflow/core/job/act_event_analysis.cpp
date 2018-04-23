#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/event_report_util.h"
namespace oneflow {

std::string GetLastUpRegstTime(
    const std::vector<int64_t>& regsts,
    const HashMap<std::string, RegstEvent>& regst_events,
    const int64_t actor_id, const int64_t act_id) {
  // key: regst_desc_id _ actor_id _ act_id
  int64_t id = -1;
  double time = 0.0;
  for (int64_t regst_desc_id : regsts) {
    std::string key = std::to_string(regst_desc_id) + "_"
                      + std::to_string(actor_id) + "_" + std::to_string(act_id);
    auto re = regst_events.find(key);
    if (re == regst_events.end()) return ",,";
    if (re->second.from_producer_time > time) {
      time = re->second.from_producer_time;
      id = regst_desc_id;
    }
  }
  return std::to_string(id) + "," + Time2String(time) + ",";
}
std::string GetLastDownRegstTime(
    const std::vector<int64_t>& regsts,
    const HashMap<std::string, RegstEvent>& regst_events,
    const int64_t actor_id, const int64_t act_id) {
  // key: regst_desc_id _ actor_id _ act_id
  int64_t id = -1;
  double time = 0.0;
  for (int64_t regst_desc_id : regsts) {
    std::string key = std::to_string(regst_desc_id) + "_"
                      + std::to_string(actor_id) + "_" + std::to_string(act_id);
    auto re = regst_events.find(key);
    if (re == regst_events.end()) return ",,";
    if (re->second.from_consumer_time > time) {
      time = re->second.from_consumer_time;
      id = regst_desc_id;
    }
  }
  return std::to_string(id) + "," + Time2String(time) + ",";
}
void ActEventAnalysis(const std::string& plan_filepath,
                      const std::string& act_event_filepath,
                      const std::string& time_diff_filepath,
                      const std::string& msg_event_filepath,
                      const std::string& report_filepath) {
  Global<JobDesc>::New();
  HashMap<int64_t, std::vector<int64_t>> actor_id2produced_regsts;
  HashMap<int64_t, std::vector<int64_t>> actor_id2consumed_regsts;
  ActorId2RegstId(plan_filepath, actor_id2produced_regsts,
                  actor_id2consumed_regsts);
  std::vector<double> machine_time_diffs;
  GetMachineTimeDiff(time_diff_filepath, machine_time_diffs);
  HashMap<std::string, RegstEvent> regst_events;
  Msg2RegstEvents(msg_event_filepath, regst_events, time_diff_filepath);
  Plan plan;
  ParseProtoFromTextFile(plan_filepath, &plan);
  std::ofstream out_stream(report_filepath);
  out_stream << "actor,type,machine,thrd,stream,act_id,push_time,start_time,"
                "stop_time,"
             << "block_time(s),block_time(ms),run_time(s),run_time(ms),"
             << "last_up_regst,read_ready,last_down_regst,write_ready,"
             << "\n";
  auto act_events = of_make_unique<std::list<ActEvent>>();
  LoadEvents<ActEvent>(act_event_filepath, act_events.get());
  for (auto event : *act_events) {
    int64_t actor_id = event.actor_id();
    double time_diff = GetDiffTime(actor_id, machine_time_diffs);
    out_stream << "'" << std::to_string(actor_id) + ",";
    out_stream << GetActorInfo(plan, actor_id) + ",";
    out_stream << "'" << std::to_string(event.work_stream_id()) + ",";
    out_stream << "'" << std::to_string(event.act_id()) + ",";
    out_stream << "'" << Time2String(event.push_time() - time_diff) + ",";
    out_stream << "'" << Time2String(event.start_time() - time_diff) + ",";
    out_stream << "'" << Time2String(event.stop_time() - time_diff) + ",";
    out_stream << Time2String(event.start_time() - event.push_time()) + ",";
    out_stream << Time2HumanReadable(event.start_time() - event.push_time())
                      + ",";
    out_stream << Time2String(event.stop_time() - event.start_time()) + ",";
    out_stream << Time2HumanReadable(event.stop_time() - event.start_time())
                      + ",";
    out_stream << GetLastUpRegstTime(actor_id2produced_regsts[actor_id],
                                     regst_events, actor_id, event.act_id());
    out_stream << GetLastDownRegstTime(actor_id2consumed_regsts[actor_id],
                                       regst_events, actor_id, event.act_id());
    out_stream << "\n";
  }
  out_stream.close();
  Global<JobDesc>::Delete();
}
}  // namespace oneflow

DEFINE_string(act_event_filepath, "act_event.bin", "");
DEFINE_string(time_diff_filepath, "time_diff.txt", "");
DEFINE_string(plan_filepath, "naive_plan", "");
DEFINE_string(msg_event_filepath, "msg_event.bin", "");
DEFINE_string(report_filepath, "act_event_analysis.csv", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "make a msg report from " << FLAGS_msg_event_filepath;
  oneflow::ActEventAnalysis(FLAGS_plan_filepath, FLAGS_act_event_filepath,
                            FLAGS_time_diff_filepath, FLAGS_msg_event_filepath,
                            FLAGS_report_filepath);
  return 0;
}
