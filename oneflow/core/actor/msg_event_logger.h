#ifndef ONEFLOW_CORE_ACTOR_MSG_EVENT_LOGGER_H_
#define ONEFLOW_CORE_ACTOR_MSG_EVENT_LOGGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/actor/msg_event.pb.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class MsgEventLogger final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MsgEventLogger);
  ~MsgEventLogger() = default;

  void PrintMsgEventToLogDir(const MsgEvent&);

  static const std::string msg_event_bin_filename_;
  static const std::string msg_event_txt_filename_;

 private:
  friend class Global<MsgEventLogger>;
  MsgEventLogger();

  PersistentOutStream bin_out_stream_;
  PersistentOutStream txt_out_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_MSG_EVENT_LOGGER_H_
