#include "oneflow/core/actor/kernel_event_logger.h"
#include "oneflow/core/common/protobuf.h"
#include <google/protobuf/text_format.h>

namespace oneflow {

const std::string KernelEventLogger::kernel_event_bin_filename_(
    "kernel_event.bin");
const std::string KernelEventLogger::kernel_event_txt_filename_(
    "kernel_event.txt");

void KernelEventLogger::PrintKernelEventToLogDir(
    const KernelEvent& kernel_event) {
  bin_out_stream_ << kernel_event;
  std::string kernel_event_txt;
  google::protobuf::TextFormat::PrintToString(kernel_event, &kernel_event_txt);
  txt_out_stream_ << kernel_event_txt;
}

KernelEventLogger::KernelEventLogger()
    : bin_out_stream_(LocalFS(),
                      JoinPath(LogDir(), kernel_event_bin_filename_)),
      txt_out_stream_(LocalFS(),
                      JoinPath(LogDir(), kernel_event_txt_filename_)) {}

}  // namespace oneflow
