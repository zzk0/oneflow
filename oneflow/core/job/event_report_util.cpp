#include "oneflow/core/common/protobuf.h"
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
void ActorId2RegstId(
    const std::string& plan_filepath,
    HashMap<int64_t, std::vector<int64_t>>& actor_id2produced_regsts,
    HashMap<int64_t, std::vector<int64_t>>& actor_id2consumed_regsts) {
  Plan plan;
  ParseProtoFromTextFile(plan_filepath, &plan);
  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& regst_desc_it : task_proto.produced_regst_desc()) {
      actor_id2produced_regsts[task_proto.task_id()].emplace_back(
          regst_desc_it.second.regst_desc_id());
    }
    for (const auto& regst_desc_it : task_proto.consumed_regst_desc_id()) {
      actor_id2consumed_regsts[task_proto.task_id()].emplace_back(
          regst_desc_it.second);
    }
  }
  /*for (const auto& v : actor_id2produced_regsts) {
    for (const auto& i : v.second) {
      std::cout << "produced_regst " << v.first << " " << i << std::endl;
    }
  }
  for (const auto& v : actor_id2consumed_regsts) {
    for (const auto& i : v.second) {
      std::cout << "consumed_regst " << v.first << " " << i << std::endl;
    }
  }*/
}

void GetMachineTimeDiff(const std::string& time_diff_filepath,
                        std::vector<double>& machine_time_diffs) {
  std::ifstream in_stream(time_diff_filepath);
  std::string line;
  machine_time_diffs.clear();
  while (std::getline(in_stream, line)) {
    machine_time_diffs.push_back(std::stod(line) * 1e6);
  }
  for (auto v : machine_time_diffs) std::cout << v << std::endl;
  in_stream.close();
}

double GetEventTime(const MsgEvent& event,
                    const std::vector<double> machine_time_diffs) {
  if (machine_time_diffs.size() == 0) {
    return event.time();
  } else {
    // int64_t machine_id = event.src_actor_id() >> (63 - machine_id_bit_num_);
    int64_t machine_id = event.src_actor_id() >> (63 - 16);
    // Global<IDMgr>::Get()->MachineId4ActorId(event.src_actor_id());
    CHECK_LT(machine_id, machine_time_diffs.size());
    double t = machine_time_diffs.at(machine_id);
    double e = event.time();
    e = t + e;
    std::cout << machine_id << ": " << Time2String(event.time()) << " "
              << Time2String(e) << " " << machine_time_diffs.at(machine_id)
              << " t=" << t << std::endl;
    return event.time() + machine_time_diffs.at(machine_id);
  }
}

double GetDiffTime(const int64_t actor_id,
                   const std::vector<double> machine_time_diffs) {
  if (machine_time_diffs.size() == 0) {
    return 0.0;
  } else {
    // int64_t machine_id = event.src_actor_id() >> (63 - machine_id_bit_num_);
    int64_t machine_id = actor_id >> (63 - 16);
    CHECK_LT(machine_id, machine_time_diffs.size());
    return machine_time_diffs.at(machine_id);
  }
}

void Msg2RegstEvents(const std::string& msg_event_filepath,
                     HashMap<std::string, RegstEvent>& regst_events,
                     const std::string& time_diff_filepath) {
  std::vector<double> machine_time_diffs;
  if (time_diff_filepath != "") {
    GetMachineTimeDiff(time_diff_filepath, machine_time_diffs);
  }
  auto msg_events = of_make_unique<std::list<MsgEvent>>();
  LoadEvents<MsgEvent>(msg_event_filepath, msg_events.get());
  std::string key;
  for (auto event : *msg_events) {
    if (event.producer_actor_id() == event.src_actor_id()) {
      key = std::to_string(event.regst_desc_id()) + "_"
            + std::to_string(event.dst_actor_id()) + "_"
            + std::to_string(event.piece_id());  // + "_"
      //+ std::to_string(event.act_id());
    } else {
      key = std::to_string(event.regst_desc_id()) + "_"
            + std::to_string(event.src_actor_id()) + "_"
            + std::to_string(event.piece_id());  // + "_"
      //+ std::to_string(event.act_id());
    }
    auto it = regst_events.find(key);
    if (it == regst_events.end()) {
      RegstEvent re;
      re.regst_desc_id = event.regst_desc_id();
      re.producer_id = event.producer_actor_id();
      re.piece_id = event.piece_id();
      re.act_id = event.act_id();
      regst_events[key] = re;
    }
    if (event.info() == "to_consumer") {
      regst_events[key].consumer_id = event.dst_actor_id();
      regst_events[key].to_consumer_time =
          event.time() - GetDiffTime(event.src_actor_id(), machine_time_diffs);
    } else if (event.info() == "from_producer") {
      CHECK_EQ(regst_events[key].consumer_id, event.dst_actor_id());
      regst_events[key].from_producer_time =
          event.time() - GetDiffTime(event.dst_actor_id(), machine_time_diffs);
    } else if (event.info() == "to_producer") {
      CHECK_EQ(regst_events[key].consumer_id, event.src_actor_id());
      regst_events[key].to_producer_time =
          event.time() - GetDiffTime(event.src_actor_id(), machine_time_diffs);
    } else if (event.info() == "from_consumer") {
      CHECK_EQ(regst_events[key].consumer_id, event.src_actor_id());
      regst_events[key].from_consumer_time =
          event.time() - GetDiffTime(event.dst_actor_id(), machine_time_diffs);
    } else {
      UNIMPLEMENTED();
    }
  }
}

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
