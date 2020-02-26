#ifndef ONEFLOW_XRT_TEST_TEST_EXECUTABLE_H_

#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/parameter.h"

#include <vector>
#include <functional>

namespace oneflow {
namespace xrt {

typedef std::function<void(const std::vector<Parameter> &,
                           const std::vector<Parameter> &)> FuncCode;

struct FuncArgumentIndices {
  std::vector<int> inputs;
  std::vector<int> outputs;
};

class TestExecutable : public Executable {
 public:
  TestExecutable(const std::string &name, const int num_inputs,
                 const std::vector<Parameter> &outputs,
                 const std::vector<Parameter> &temp_buffers,
                 const std::vector<FuncCode> &func_codes,
                 const std::vector<FuncArgumentIndices> &func_args);

  bool Run(const std::vector<Parameter> &inputs,
           const ExecutableRunOptions &run_options,
           bool block_until_done = true) override;

 private:
  int num_inputs_;
  std::vector<Parameter> outputs_;
  std::vector<Parameter> temp_buffers_;
  std::vector<FuncCode> func_codes_;
  std::vector<FuncArgumentIndices> func_args_;
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TEST_TEST_EXECUTABLE_H_
