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
#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/job/placement.cfg.h"

namespace py = pybind11;

namespace oneflow {

namespace one {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<MirroredTensor, std::shared_ptr<MirroredTensor>>(m, "Tensor")
      .def(
          py::init([](const py::tuple& py_shape, int dtype, const std::shared_ptr<Device>& device) {
            DimVector shape_dims;
            CHECK(py::isinstance<py::tuple>(py_shape));
            for (auto dim : py_shape) { shape_dims.emplace_back(dim.cast<int64_t>()); }
            std::shared_ptr<Shape> shape = std::make_shared<Shape>(shape_dims);
            return std::make_shared<MirroredTensor>(shape, static_cast<DataType>(dtype), device);
          }))
      .def(py::init([](const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                       const std::shared_ptr<compatible_py::Distribute>& distribute) {
        return std::make_shared<MirroredTensor>(lbi, job_name, distribute);
      }))
      .def(py::init([](const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                       const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                       const std::shared_ptr<compatible_py::BlobRegister>& blob_register,
                       const std::string& job_name,
                       const std::shared_ptr<compatible_py::Distribute>& distribute) {
        return std::make_shared<MirroredTensor>(lbi, blob_object, blob_register, job_name,
                                                distribute);
      }))
      .def_property_readonly("parallel_conf", &MirroredTensor::parallel_conf)
      .def_property_readonly("shape", &MirroredTensor::shape)
      .def("get_dtype",
           [](std::shared_ptr<MirroredTensor>& x) { return static_cast<int>(x->dtype()); })
      .def("storage", &MirroredTensor::storage)
      .def("size", &MirroredTensor::shape)
      .def_property_readonly("lbi", &MirroredTensor::lbi)
      .def_property_readonly("logical_blob_name", &MirroredTensor::logical_blob_name)
      .def_property_readonly("op_name", &MirroredTensor::op_name)
      .def_property_readonly("blob_name", &MirroredTensor::blob_name)
      .def_property_readonly("batch_axis", &MirroredTensor::batch_axis)
      .def_property_readonly("is_dynamic", &MirroredTensor::is_dynamic)
      .def_property_readonly("is_tensor_list", &MirroredTensor::is_tensor_list)
      .def_property_readonly("distribute", &MirroredTensor::distribute)
      .def_property_readonly("unique_name", &MirroredTensor::unique_name)
      .def_property_readonly("job_name", &MirroredTensor::job_name)
      .def_property_readonly("parallel_size", &MirroredTensor::parallel_size)
      .def_property_readonly("blob_object", &MirroredTensor::blob_object)
      .def("set_job_name", &MirroredTensor::set_job_name);

  py::class_<ConsistentTensor, std::shared_ptr<ConsistentTensor>>(m, "ConsistentTensor")
      .def(py::init([](const std::shared_ptr<Shape>& shape, DataType dtype,
                       const std::shared_ptr<compatible_py::Distribute>& distribute,
                       std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
        return std::make_shared<ConsistentTensor>(shape, dtype, distribute, parallel_conf);
      }))
      .def(py::init([](const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                       const std::shared_ptr<compatible_py::Distribute>& distribute) {
        return std::make_shared<ConsistentTensor>(lbi, job_name, distribute);
      }))
      .def(py::init([](const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                       const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                       const std::shared_ptr<compatible_py::BlobRegister>& blob_register,
                       const std::string& job_name,
                       const std::shared_ptr<compatible_py::Distribute>& distribute) {
        return std::make_shared<ConsistentTensor>(lbi, blob_object, blob_register, job_name,
                                                  distribute);
      }))
      .def_property_readonly("parallel_conf", &ConsistentTensor::parallel_conf)
      .def_property_readonly("shape", &ConsistentTensor::shape)
      .def("get_dtype",
           [](std::shared_ptr<ConsistentTensor>& x) { return static_cast<int>(x->dtype()); });
}

}  // namespace one

}  // namespace oneflow
