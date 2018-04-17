#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/event_report_util.h"

namespace oneflow {

void MsgEventReport(/*const std::string& plan_filepath,*/
                    const std::string& msg_event_filepath,
                    const std::string& report_filepath) {
  Global<JobDesc>::New();
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

DEFINE_string(msg_event_filepath, "msg_event.bin", "");
DEFINE_string(report_filepath, "msg_event.csv", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "make a msg report from " << FLAGS_msg_event_filepath;
  oneflow::MsgEventReport(/*FLAGS_plan_filepath, */ FLAGS_msg_event_filepath,
                          FLAGS_report_filepath);
  return 0;
}
