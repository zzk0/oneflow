#ifndef ONEFLOW_XRT_TEST_TEST_OP_KERNEL_H_
#define ONEFLOW_XRT_TEST_TEST_OP_KERNEL_H_

#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/graph/node.h"

#include "oneflow/xrt/kernel/op_kernel.h"

#include <functional>
#include <unordered_map>
#include <vector>

namespace oneflow {
namespace xrt {

class TestOpContext {
 public:
  TestOpContext(const XrtNode *node,
                const std::unordered_map<std::string, Parameter> &all_params)
      : node_(node), all_params_(all_params) {}

 public:
  const XrtNode *node_;
  const std::unordered_map<std::string, Parameter> &all_params_;

  std::function<void(const std::vector<Parameter> &,
                     const std::vector<Parameter> &)> func_code_;
  std::vector<std::string> input_args_;
  std::vector<std::string> output_args_;
  std::unordered_map<std::string, Parameter> tmp_buffers_;
};

class TestOpKernel : public OpKernel<TestOpContext> {
 public:
  virtual void Compile(TestOpContext *ctx) = 0;
};

#define REGISTER_TEST_OP_KERNEL(OpName, KernelType)                 \
  static auto _test_op_kernel_##OpName##_ __attribute__((unused)) = \
      OpKernelRegistrar<TestOpContext>(#OpName)                     \
          .SetField(XrtEngine::TEST)                                \
          .SetDevice({XrtDevice::GPU_CUDA})                         \
          .SetFactory([]() -> OpKernel<TestOpContext> * {           \
                          return new KernelType;                    \
                      })

inline std::shared_ptr<OpKernel<TestOpContext>> BuildTestOpKernel(
    const std::string &op_type) {
  auto field = MakeXrtField(XrtDevice::GPU_CUDA, XrtEngine::TEST);
  return std::shared_ptr<OpKernel<TestOpContext>>(
      OpKernelBuilder<TestOpContext>()(field, op_type));
}

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TEST_TEST_OP_KERNEL_H_
