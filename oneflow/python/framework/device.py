"""
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
"""
import oneflow_api
import oneflow.core.common.device_type_pb2 as device_type_pb2
from oneflow.python.oneflow_export import oneflow_export

_STR_DEVICETYPE2PROTO_DEVICETYPE = {
    "cuda": device_type_pb2.kGPU,
    "cpu": device_type_pb2.kCPU,
}


@oneflow_export("device")
class Device(oneflow_api.device):
    def __init__(self, device: str):
        device_list = device.split(":")
        if len(device_list) > 2:
            raise RuntimeError("Invalid device string: ", device)
        device_type_str = device_list[0]
        if device_type_str in _STR_DEVICETYPE2PROTO_DEVICETYPE:
            device_type = _STR_DEVICETYPE2PROTO_DEVICETYPE[device_type_str]
        else:
            raise RuntimeError(
                "Expected one of cpu, cuda device type at start of device string "
                + device_type_str,
            )
        if len(device_list) > 1:
            device_index = int(device_list[1])
            if device_type_str == "cpu" and device_index != 0:
                raise RuntimeError("CPU device index must be 0")
        else:
            device_index = 0
        oneflow_api.device.__init__(self, device_type, device_index)
