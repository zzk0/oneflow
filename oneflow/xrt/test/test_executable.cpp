#include "oneflow/xrt/test/test_executable.h"

namespace oneflow {
namespace xrt {

TestExecutable::TestExecutable(const std::string &name, const int num_inputs,
                               const std::vector<Parameter> &outputs,
                               const std::vector<Parameter> &temp_buffers,
                               const std::vector<FuncCode> &func_codes,
                               const std::vector<FuncArgumentIndices> &func_args)
    : Executable(name, XrtEngine::TEST),
      num_inputs_(num_inputs),
      outputs_(outputs),
      temp_buffers_(temp_buffers),
      func_codes_(func_codes),
      func_args_(func_args) {}

bool TestExecutable::Run(const std::vector<Parameter> &inputs,
                         const ExecutableRunOptions &run_options,
                         bool block_until_done) {
  auto PullArgs = [&](const std::vector<int> &indices) {
    std::vector<Parameter> args;
    for (int idx : indices) {
      if (idx < num_inputs_) {
        args.push_back(inputs[idx]);
      } else if (idx < num_inputs_ + outputs_.size()) {
        args.push_back(outputs_[idx - num_inputs_]);
      } else {
        idx -= (num_inputs_ + outputs_.size());
        CHECK_GE(idx, 0);
        CHECK_LT(idx, temp_buffers_.size());
        args.push_back(temp_buffers_[idx]);
      }
    }
    return std::move(args);
  };

  CHECK_EQ(inputs.size(), num_inputs_);

  for (int i = 0; i < func_codes_.size(); ++i) {
    auto in_args = PullArgs(func_args_[i].inputs);
    auto out_args = PullArgs(func_args_[i].outputs);
    func_codes_[i](in_args, out_args);
  }

  // Synchronize stream if block_until_done.
  if (block_until_done) {
    // TODO(hjchen2)
  }

  // All return params are the results of the executable.
  this->results_ = run_options.return_params;
  return true /*running status*/;
}

}  // namespace xrt
}  // namespace oneflow
