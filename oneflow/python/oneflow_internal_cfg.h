/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <stdint.h>
#include "oneflow/python/oneflow_internal_helper.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {
namespace oneflow_api {

std::shared_ptr<::oneflow::cfg::ErrorProto> RegisterForeignCallbackOnlyOnce(
    oneflow::ForeignCallback* callback) {
  return oneflow::RegisterForeignCallbackOnlyOnce(callback).GetDataAndErrorProto();
}

std::shared_ptr<::oneflow::cfg::ErrorProto> RegisterWatcherOnlyOnce(oneflow::ForeignWatcher* watcher) {
  return oneflow::RegisterWatcherOnlyOnce(watcher).GetDataAndErrorProto();
}

std::pair<bool, std::shared_ptr<::oneflow::cfg::ErrorProto>> IsOpTypeCaseCpuSupportOnly(int64_t op_type_case) {
  return oneflow::IsOpTypeCaseCpuSupportOnly(op_type_case).GetDataAndErrorProto(false);
}

std::pair<bool, std::shared_ptr<::oneflow::cfg::ErrorProto>> IsOpTypeNameCpuSupportOnly(
    const std::string& op_type_name) {
  return oneflow::IsOpTypeNameCpuSupportOnly(op_type_name).GetDataAndErrorProto(false);
}

std::pair<std::string, std::shared_ptr<::oneflow::cfg::ErrorProto>> CurrentResource() {
  return oneflow::CurrentResource().GetDataAndErrorProto(std::string(""));
}

std::pair<std::string, std::shared_ptr<::oneflow::cfg::ErrorProto>> EnvResource() {
  return oneflow::EnvResource().GetDataAndErrorProto(std::string(""));
}

void EnableEagerEnvironment(bool enable_eager_execution) {
  using namespace oneflow;
  *Global<bool, EagerExecution>::Get() = enable_eager_execution;
}

bool IsEnvInited() {
  using namespace oneflow;
  return Global<EnvGlobalObjectsScope>::Get() != nullptr;
}

std::shared_ptr<::oneflow::cfg::ErrorProto> InitEnv(const std::string& env_proto_str) {
  return oneflow::InitEnv(env_proto_str).GetDataAndErrorProto();
}

std::shared_ptr<::oneflow::cfg::ErrorProto> DestroyEnv() {
  return oneflow::DestroyEnv().GetDataAndErrorProto();
}

bool IsSessionInited() {
  using namespace oneflow;
  return Global<SessionGlobalObjectsScope>::Get() != nullptr;
}

std::shared_ptr<::oneflow::cfg::ErrorProto> InitGlobalSession(const std::string& config_proto_str) {
//   using namespace oneflow;
  return ::oneflow::InitGlobalSession(config_proto_str).GetDataAndErrorProto();
}

std::shared_ptr<::oneflow::cfg::ErrorProto> DestroyGlobalSession() {
  return oneflow::DestroyGlobalSession().GetDataAndErrorProto();
}

std::shared_ptr<::oneflow::cfg::ErrorProto> StartGlobalSession() {
  return oneflow::StartGlobalSession().GetDataAndErrorProto();
}

std::shared_ptr<::oneflow::cfg::ErrorProto> StopGlobalSession() {
  return oneflow::StopGlobalSession().GetDataAndErrorProto();
}

std::pair<std::string, std::shared_ptr<::oneflow::cfg::ErrorProto>> GetSerializedInterUserJobInfo() {
  return oneflow::GetSerializedInterUserJobInfo().GetDataAndErrorProto(std::string(""));
}

std::pair<std::string, std::shared_ptr<::oneflow::cfg::ErrorProto>> GetSerializedJobSet() {
  return oneflow::GetSerializedJobSet().GetDataAndErrorProto(std::string(""));
}

std::pair<std::string, std::shared_ptr<::oneflow::cfg::ErrorProto>> GetSerializedStructureGraph() {
  return oneflow::GetSerializedStructureGraph().GetDataAndErrorProto(std::string(""));
}

std::pair<std::string, std::shared_ptr<::oneflow::cfg::ErrorProto>> GetFunctionConfigDef() {
  return oneflow::GetFunctionConfigDef().GetDataAndErrorProto(std::string(""));
}

std::shared_ptr<::oneflow::cfg::ErrorProto> LaunchJob(const std::shared_ptr<oneflow::ForeignJobInstance>& cb) {
  return oneflow::LaunchJob(cb).GetDataAndErrorProto();
}

std::pair<std::string, std::shared_ptr<::oneflow::cfg::ErrorProto>>
GetMachine2DeviceIdListOFRecordFromParallelConf(const std::string& parallel_conf) {
  return oneflow::GetSerializedMachineId2DeviceIdListOFRecord(parallel_conf)
      .GetDataAndErrorProto(std::string(""));
}

std::pair<long, std::shared_ptr<::oneflow::cfg::ErrorProto>> GetUserOpAttrType(const std::string& op_type_name,
                                                                   const std::string& attr_name) {
  return oneflow::GetUserOpAttrType(op_type_name, attr_name).GetDataAndErrorProto(0LL);
}

std::pair<std::string, std::shared_ptr<::oneflow::cfg::ErrorProto>> InferOpConf(
    const std::string& serialized_op_conf, const std::string& serialized_op_input_signature) {
  return oneflow::InferOpConf(serialized_op_conf, serialized_op_input_signature)
      .GetDataAndErrorProto(std::string(""));
}

bool IsInterfaceOpTypeCase(int64_t op_type_case) {
  return oneflow::IsClassRegistered<oneflow::IsInterfaceOpConf4OpTypeCase>(op_type_case);
}

std::pair<long, std::shared_ptr<::oneflow::cfg::ErrorProto>> GetOpParallelSymbolId(
    const std::string& serialized_op_conf) {
  return oneflow::GetOpParallelSymbolId(serialized_op_conf).GetDataAndErrorProto(0LL);
}

std::pair<std::string, std::shared_ptr<::oneflow::cfg::ErrorProto>> CheckAndCompleteUserOpConf(
    const std::string& serialized_op_conf) {
  return oneflow::CheckAndCompleteUserOpConf(serialized_op_conf)
      .GetDataAndErrorProto(std::string(""));
}

std::shared_ptr<::oneflow::cfg::ErrorProto> RunLogicalInstruction(const std::string& vm_instruction_list,
                                                       const std::string& eager_symbol_list_str) {
  return oneflow::RunLogicalInstruction(vm_instruction_list, eager_symbol_list_str)
      .GetDataAndErrorProto();
}

std::shared_ptr<::oneflow::cfg::ErrorProto> RunPhysicalInstruction(const std::string& vm_instruction_list,
                                                        const std::string& eager_symbol_list_str) {
  return oneflow::RunPhysicalInstruction(vm_instruction_list, eager_symbol_list_str)
      .GetDataAndErrorProto();
}

std::pair<long, std::shared_ptr<::oneflow::cfg::ErrorProto>> CurrentMachineId() {
  return oneflow::CurrentMachineId().GetDataAndErrorProto(0LL);
}

std::pair<long, std::shared_ptr<::oneflow::cfg::ErrorProto>> NewLogicalObjectId() {
  return oneflow::NewLogicalObjectId().GetDataAndErrorProto(0LL);
}

std::pair<long, std::shared_ptr<::oneflow::cfg::ErrorProto>> NewLogicalSymbolId() {
  return oneflow::NewLogicalSymbolId().GetDataAndErrorProto(0LL);
}

std::pair<long, std::shared_ptr<::oneflow::cfg::ErrorProto>> NewPhysicalObjectId() {
  return oneflow::NewPhysicalObjectId().GetDataAndErrorProto(0LL);
}

std::pair<long, std::shared_ptr<::oneflow::cfg::ErrorProto>> NewPhysicalSymbolId() {
  return oneflow::NewPhysicalSymbolId().GetDataAndErrorProto(0LL);
}

int Ofblob_GetDataType(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->data_type();
}

size_t OfBlob_NumAxes(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->NumAxes();
}

void OfBlob_CopyShapeFromNumpy(uint64_t of_blob_ptr, long* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyShapeFrom(array, size);
}

void OfBlob_CopyShapeToNumpy(uint64_t of_blob_ptr, long* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyShapeTo(array, size);
}

bool OfBlob_IsDynamic(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->is_dynamic();
}

bool OfBlob_IsTensorList(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->is_tensor_list();
}

long OfBlob_TotalNumOfTensors(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->TotalNumOfTensors();
}

long OfBlob_NumOfTensorListSlices(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->NumOfTensorListSlices();
}

long OfBlob_TensorIndex4SliceId(uint64_t of_blob_ptr, int32_t slice_id) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->TensorIndex4SliceId(slice_id);
}

void OfBlob_AddTensorListSlice(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->AddTensorListSlice();
}

void OfBlob_ResetTensorIterator(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->ResetTensorIterator();
}

void OfBlob_IncTensorIterator(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->IncTensorIterator();
}

bool OfBlob_CurTensorIteratorEqEnd(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CurTensorIteratorEqEnd();
}

void OfBlob_CopyStaticShapeTo(uint64_t of_blob_ptr, long* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyStaticShapeTo(array, size);
}

void OfBlob_CurTensorCopyShapeTo(uint64_t of_blob_ptr, long* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CurTensorCopyShapeTo(array, size);
}

void OfBlob_ClearTensorLists(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->ClearTensorLists();
}

void OfBlob_AddTensor(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->AddTensor();
}

bool OfBlob_CurMutTensorAvailable(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CurMutTensorAvailable();
}

void OfBlob_CurMutTensorCopyShapeFrom(uint64_t of_blob_ptr, long* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CurMutTensorCopyShapeFrom(array, size);
}

std::shared_ptr<::oneflow::cfg::ErrorProto> CacheInt8Calibration() {
  return oneflow::CacheInt8Calibration().GetDataAndErrorProto();
}

std::shared_ptr<::oneflow::cfg::ErrorProto> WriteInt8Calibration(const std::string& path) {
  return oneflow::WriteInt8Calibration(path).GetDataAndErrorProto();
}

} // namespace oneflow_api

} // namespace oneflow
