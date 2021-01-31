#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/framework/tensor.h"

namespace py = pybind11;

namespace oneflow {

namespace one {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<MirroredTensor, std::shared_ptr<MirroredTensor>>(m, "Tensor")
    .def(py::init(
      [](const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
               const std::shared_ptr<compatible_py::Distribute>& distribute) {
      return std::make_shared<MirroredTensor>(lbi, job_name, distribute);
    }))
    .def(py::init(
     [](const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                 const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                 const std::shared_ptr<compatible_py::BlobRegister>& blob_register, const std::string& job_name,
                 const std::shared_ptr<compatible_py::Distribute>& distribute){
      return std::make_shared<MirroredTensor>(lbi, blob_object, blob_register, job_name, distribute);
    }))
    .def_property_readonly("lbi", &MirroredTensor::lbi)
    .def_property_readonly("logical_blob_name", &MirroredTensor::logical_blob_name)
    .def_property_readonly("op_name", &MirroredTensor::op_name)
    .def_property_readonly("blob_name", &MirroredTensor::blob_name)
    .def_property_readonly("shape", &MirroredTensor::shape)
    .def_property_readonly("dtype", &MirroredTensor::dtype)
    .def_property_readonly("batch_axis", &MirroredTensor::batch_axis)
    .def_property_readonly("is_dynamic", &MirroredTensor::is_dynamic)
    .def_property_readonly("is_tensor_list", &MirroredTensor::is_tensor_list)
    .def_property_readonly("parallel_conf", &MirroredTensor::parallel_conf)
    .def_property_readonly("distribute", &MirroredTensor::distribute)
    .def_property_readonly("unique_name", &MirroredTensor::unique_name);


}

}
}

