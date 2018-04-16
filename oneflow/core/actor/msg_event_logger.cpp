#include "oneflow/core/actor/msg_event_logger.h"
#include "oneflow/core/common/protobuf.h"
#include <google/protobuf/text_format.h>

namespace oneflow {

const std::string MsgEventLogger::msg_event_bin_filename_("msg_event.bin");
const std::string MsgEventLogger::msg_event_txt_filename_("msg_event.txt");

void MsgEventLogger::PrintMsgEventToLogDir(const MsgEvent& msg_event) {
  bin_out_stream_ << msg_event;
  std::string msg_event_txt;
  google::protobuf::TextFormat::PrintToString(msg_event, &msg_event_txt);
  txt_out_stream_ << msg_event_txt;
}

MsgEventLogger::MsgEventLogger()
    : bin_out_stream_(LocalFS(), JoinPath(LogDir(), msg_event_bin_filename_)),
      txt_out_stream_(LocalFS(), JoinPath(LogDir(), msg_event_txt_filename_)) {}

}  // namespace oneflow
