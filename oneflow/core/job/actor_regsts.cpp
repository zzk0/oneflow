#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/event_report_util.h"
namespace oneflow {
void Map2Csv(const std::string& filepath, const HashMap<int64_t, std::vector<int64_t>& actor_id2regsts) {
  std::ofstream out_stream(filepath);
  out_stream << "actor,regsts.../"
             << "\n";
  for (auto  pair: actor_id2produced_regsts) {
    out_stream << "'" << std::to_string(pair->first) <<  ",";
    for (auto regst : pair->second) { 
      out_stream << std::to_string(regst) << ",";
    }
    out_stream << "\n";
  }
  out_stream.close();
}
void ActorRegsts(const std::string& plan_filepath,
                      const std::string& report_prefix) {
  Global<JobDesc>::New();
  HashMap<int64_t, std::vector<int64_t>> actor_id2produced_regsts;
  HashMap<int64_t, std::vector<int64_t>> actor_id2consumed_regsts;
  ActorId2RegstId(plan_filepath, actor_id2produced_regsts,
                  actor_id2consumed_regsts);
  Map2Csv(JoinPath(report_folder, "produced_regsts.csv"), actor_id2produced_regsts);
  Map2Csv(JoinPath(report_folder, "consumed_regsts.csv"), actor_id2consumed_regsts);
  Global<JobDesc>::Delete();
}
}  // namespace oneflow

DEFINE_string(plan_filepath, "naive_plan", "");
DEFINE_string(report_folder, "./output", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "make a msg report from " << FLAGS_msg_event_filepath;
  oneflow::ActorRegsts(FLAGS_plan_filepath, FLAGS_report_folder);
  return 0;
}
