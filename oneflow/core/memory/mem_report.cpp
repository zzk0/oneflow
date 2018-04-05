#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {
std::string GetExecNodeName(const TaskProto& task) {
  std::string name = "[";
  int n = 0;
  for (const auto& exec_node : task.exec_sequence().exec_node()) {
    name = name + exec_node.kernel_conf().op_conf().name() + " ";
    ++n;
    if (n > 2) {
      name = name + "... ";
      break;
    }
  }
  if (name.length() > 1) name.pop_back();
  name = name + "]";
  return name;
}

std::string Double2String(double d) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(1) << d;
  return stream.str();
}
std::string GetRegstMemSize(const RegstDescProto& regst_desc_proto) {
  const RtRegstDesc* runtime_regst_desc = new RtRegstDesc(regst_desc_proto);
  size_t size = runtime_regst_desc->packed_blob_desc()->TotalByteSize();
  double ds = size;
  ds = ds / 1024.0;
  if (ds < 1024.0) return Double2String(ds) + "KB," + std::to_string(size);
  ds = ds / 1024.0;
  if (ds < 1024.0) return Double2String(ds) + "MB," + std::to_string(size);
  ds = ds / 1024.0;
  return Double2String(ds) + "GB," + std::to_string(size);
}
std::vector<std::string> GetProducedRegstDescInfo(const TaskProto& task) {
  std::vector<std::string> infos;
  std::string info;
  for (const auto& pair : task.produced_regst_desc()) {
    info = pair.first + ",";
    info = info + std::to_string(pair.second.regst_desc_id()) + ",";
    if (pair.second.mem_case().has_device_cuda_mem()) {
      info =
          info + "GPU:"
          + std::to_string(pair.second.mem_case().device_cuda_mem().device_id())
          + ",";
    } else if (pair.second.mem_case().has_host_pinned_mem())
      info = info + "host_pinned,";
    else if (pair.second.mem_case().has_host_pageable_mem())
      info = info + "host_pageable,";
    else
      info = info + "unknown,";
    info = info + std::to_string(pair.second.register_num()) + ",";
    info = info + GetRegstMemSize(pair.second);
    infos.push_back(info);
  }
  return infos;
}
void MemReport(const std::string& plan_filepath, std::string& report_filepath) {
  Global<JobDesc>::New();
  Plan plan;
  ParseProtoFromTextFile(plan_filepath, &plan);
  PersistentOutStream out_stream(LocalFS(), report_filepath);
  out_stream << "task_type,machine_id,thrd_id,task_id,ExecNodes,";
  out_stream << "reg_key,regst_desc_id,mem_case,regst_num,size,size\n";
  std::string row;
  for (const TaskProto& task : plan.task()) {
    row = TaskType_Name(task.task_type()) + ","
          + std::to_string(task.machine_id()) + ","
          + std::to_string(task.thrd_id()) + "," + "'"
          + std::to_string(task.task_id()) + "," + GetExecNodeName(task) + ",";
    std::vector<std::string> RegstInfos = GetProducedRegstDescInfo(task);
    bool has_regst = false;
    std::string bak = row;
    for (const std::string info : RegstInfos) {
      row = row + info + "\n";
      row = row + bak;
      has_regst = true;
    }
    if (has_regst)
      row.erase(row.end() - bak.length(), row.end());
    else
      row = row + "\n";
    out_stream << row;
  }
  Global<JobDesc>::Delete();
}
}  // namespace oneflow

DEFINE_string(plan_filepath, "naive_plan", "");
DEFINE_string(report_filepath, "mem_report.csv", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "make a memory report from " << FLAGS_plan_filepath;
  oneflow::MemReport(FLAGS_plan_filepath, FLAGS_report_filepath);
  return 0;
}
