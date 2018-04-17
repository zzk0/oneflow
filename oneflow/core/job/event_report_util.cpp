#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/job/event_report_util.h"

namespace oneflow {

template<typename EventType>
void LoadEvents(const std::string& event_filepath,
                std::list<EventType>* events) {
  NormalPersistentInStream in_stream(LocalFS(), event_filepath);
  size_t event_size;
  while (
      !in_stream.Read(reinterpret_cast<char*>(&event_size), sizeof(size_t))) {
    std::vector<char> buffer(event_size);
    CHECK(!in_stream.Read(buffer.data(), event_size));
    events->emplace_back();
    events->back().ParseFromArray(buffer.data(), event_size);
  }
}

template void LoadEvents<MsgEvent>(const std::string& event_filepath,
                                   std::list<MsgEvent>* events);

template void LoadEvents<ActEvent>(const std::string& event_filepath,
                                   std::list<ActEvent>* events);

template void LoadEvents<KernelEvent>(const std::string& event_filepath,
                                      std::list<KernelEvent>* events);

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
}  // namespace oneflow
