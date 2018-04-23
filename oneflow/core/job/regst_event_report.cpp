#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/event_report_util.h"

namespace oneflow {

void RegstEventReport(/*const std::string& plan_filepath,*/
                      const std::string& msg_event_filepath,
                      const std::string& report_filepath,
                      const std::string& time_diff_filepath) {
  Global<JobDesc>::New();
  HashMap<std::string, RegstEvent> regst_events;
  Msg2RegstEvents(msg_event_filepath, regst_events, time_diff_filepath);
  std::ofstream out_stream(report_filepath);
  out_stream << "regst_desc_id,producer_id,consumer_id,act_id,piece_id,to_"
                "consumer_time,"
                "from_producer_time,to_producer_time,from_consumer_time\n";
  for (auto event : regst_events) {
    out_stream << std::to_string(event.second.regst_desc_id) + ",";
    out_stream << "'" << std::to_string(event.second.producer_id) + ",";
    out_stream << "'" << std::to_string(event.second.consumer_id) + ",";
    out_stream << std::to_string(event.second.act_id) + ",";
    out_stream << std::to_string(event.second.piece_id) + ",";
    out_stream << "'" << Time2String(event.second.to_consumer_time) + ",";
    out_stream << "'" << Time2String(event.second.from_producer_time) + ",";
    out_stream << "'" << Time2String(event.second.to_producer_time) + ",";
    out_stream << "'" << Time2String(event.second.from_consumer_time) + ",";
    out_stream << "\n";
  }
  out_stream.close();
  Global<JobDesc>::Delete();
}
}  // namespace oneflow

DEFINE_string(msg_event_filepath, "msg_event.bin", "");
DEFINE_string(report_filepath, "regst_event.csv", "");
DEFINE_string(time_diff_filepath, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "make a regst event report from " << FLAGS_msg_event_filepath;
  oneflow::RegstEventReport(/*FLAGS_plan_filepath, */ FLAGS_msg_event_filepath,
                            FLAGS_report_filepath, FLAGS_time_diff_filepath);
  return 0;
}
