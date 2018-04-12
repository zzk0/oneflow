#ifndef ONEFLOW_CORE_ACTOR_KERNEL_EVENT_LOGGER_H_
#define ONEFLOW_CORE_ACTOR_KERNEL_EVENT_LOGGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/actor/kernel_event.pb.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class KernelEventLogger final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelEventLogger);
  ~KernelEventLogger() = default;

  void PrintKernelEventToLogDir(const KernelEvent&);

  static const std::string kernel_event_bin_filename_;
  static const std::string kernel_event_txt_filename_;

 private:
  friend class Global<KernelEventLogger>;
  KernelEventLogger();

  PersistentOutStream bin_out_stream_;
  PersistentOutStream txt_out_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_KERNEL_EVENT_LOGGER_H_
