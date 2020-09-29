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
#include "oneflow/api/python/util/of_api_registry.h"
#include "oneflow/python/oneflow_internal_cfg.h"

ONEFLOW_API_PYBIND11_MODULE("", m) {

  // RegisterForeignCallbackOnlyOnce
  // RegisterWatcherOnlyOnce
  m.def("IsOpTypeCaseCpuSupportOnly", &::oneflow::oneflow_api::IsOpTypeCaseCpuSupportOnly);
  // IsOpTypeNameCpuSupportOnly
  m.def("CurrentResource", &::oneflow::oneflow_api::CurrentResource);
  m.def("EnvResource", &::oneflow::oneflow_api::EnvResource);
  m.def("EnableEagerEnvironment", &::oneflow::oneflow_api::EnableEagerEnvironment);
  m.def("IsEnvInited", &::oneflow::oneflow_api::IsEnvInited);
  // InitEnv
  m.def("DestroyEnv", &::oneflow::oneflow_api::DestroyEnv);
  m.def("IsSessionInited", &::oneflow::oneflow_api::IsSessionInited);
  // InitGlobalSession
  m.def("DestroyGlobalSession", &::oneflow::oneflow_api::DestroyGlobalSession);
  m.def("StartGlobalSession", &::oneflow::oneflow_api::StartGlobalSession);
  m.def("StopGlobalSession", &::oneflow::oneflow_api::StopGlobalSession);
  m.def("GetSerializedInterUserJobInfo", &::oneflow::oneflow_api::GetSerializedInterUserJobInfo);
  m.def("GetSerializedJobSet", &::oneflow::oneflow_api::GetSerializedJobSet);
  m.def("GetSerializedStructureGraph", &::oneflow::oneflow_api::GetSerializedStructureGraph);
  m.def("GetFunctionConfigDef", &::oneflow::oneflow_api::GetFunctionConfigDef);
  // LaunchJob
  // GetMachine2DeviceIdListOFRecordFromParallelConf
  // GetUserOpAttrType
  // InferOpConf
  m.def("IsInterfaceOpTypeCase", &::oneflow::oneflow_api::IsInterfaceOpTypeCase);
  // GetOpParallelSymbolId
  // CheckAndCompleteUserOpConf
  // RunLogicalInstruction
  // RunPhysicalInstruction
  m.def("CurrentMachineId", &::oneflow::oneflow_api::CurrentMachineId);
  m.def("NewLogicalObjectId", &::oneflow::oneflow_api::NewLogicalObjectId);
  m.def("NewLogicalSymbolId", &::oneflow::oneflow_api::NewLogicalSymbolId);
  m.def("NewPhysicalObjectId", &::oneflow::oneflow_api::NewPhysicalObjectId);
  m.def("NewPhysicalSymbolId", &::oneflow::oneflow_api::NewPhysicalSymbolId);
  m.def("Ofblob_GetDataType", &::oneflow::oneflow_api::Ofblob_GetDataType);
  m.def("OfBlob_NumAxes", &::oneflow::oneflow_api::OfBlob_NumAxes);
  // OfBlob_CopyShapeFromNumpy
  // OfBlob_CopyShapeToNumpy
  m.def("OfBlob_IsDynamic", &::oneflow::oneflow_api::OfBlob_IsDynamic);
  m.def("OfBlob_IsTensorList", &::oneflow::oneflow_api::OfBlob_IsTensorList);
  m.def("OfBlob_TotalNumOfTensors", &::oneflow::oneflow_api::OfBlob_TotalNumOfTensors);
  m.def("OfBlob_NumOfTensorListSlices", &::oneflow::oneflow_api::OfBlob_NumOfTensorListSlices);
  m.def("OfBlob_TensorIndex4SliceId", &::oneflow::oneflow_api::OfBlob_TensorIndex4SliceId);
  m.def("OfBlob_AddTensorListSlice", &::oneflow::oneflow_api::OfBlob_AddTensorListSlice);
  m.def("OfBlob_ResetTensorIterator", &::oneflow::oneflow_api::OfBlob_ResetTensorIterator);
  m.def("OfBlob_IncTensorIterator", &::oneflow::oneflow_api::OfBlob_IncTensorIterator);
  m.def("OfBlob_CurTensorIteratorEqEnd", &::oneflow::oneflow_api::OfBlob_CurTensorIteratorEqEnd);
  // OfBlob_CopyStaticShapeTo
  // OfBlob_CurTensorCopyShapeTo
  m.def("OfBlob_ClearTensorLists", &::oneflow::oneflow_api::OfBlob_ClearTensorLists);
  m.def("OfBlob_AddTensor", &::oneflow::oneflow_api::OfBlob_AddTensor);
  m.def("OfBlob_CurMutTensorAvailable", &::oneflow::oneflow_api::OfBlob_CurMutTensorAvailable);
  // OfBlob_CurMutTensorCopyShapeFrom
  m.def("CacheInt8Calibration", &::oneflow::oneflow_api::CacheInt8Calibration);
  // WriteInt8Calibration

}
