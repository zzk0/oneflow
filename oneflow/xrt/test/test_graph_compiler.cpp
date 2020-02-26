#include "oneflow/xrt/test/test_graph_compiler.h"
#include "oneflow/xrt/test/test_executable.h"
#include "oneflow/xrt/test/test_op_kernel.h"

namespace oneflow {
namespace xrt {

// Register a new graph compiler for TEST engine.
REGISTER_GRAPH_COMPILER(XrtEngine::TEST, TestGraphCompiler);

std::shared_ptr<Executable> TestGraphCompiler::Compile(
    const XrtGraph *graph,
    const std::vector<Parameter> &entry_params,
    const std::vector<Parameter> &return_params,
    const std::vector<InputOutputAlias> &aliases) {
  std::vector<Parameter> temp_buffers;
  std::vector<FuncCode> func_codes;
  std::vector<FuncArgumentIndices> func_args;

  std::unordered_map<std::string, int> indices;
  std::unordered_map<std::string, Parameter> all_params;
  for (auto param : entry_params) {
    indices.emplace(param.name(), indices.size());
    all_params[param.name()] = param;
  }
  for (auto param : return_params) {
    indices.emplace(param.name(), indices.size());
    all_params[param.name()] = param;
  }

  algorithm::TopologyVisit(*graph, [&](const XrtNode *node) {
    if (node->type() == "Argument") {
      // Argument node is not computation node, so skip it.
      return;
    }

    TestOpContext op_context(node, all_params);
    auto op_kernel = BuildTestOpKernel(node->type());
    op_kernel->Compile(&op_context);

    func_codes.push_back(op_context.func_code_);

    const auto &buffers = op_context.tmp_buffers_;
    for (auto it = buffers.begin(); it != buffers.end(); ++it) {
      all_params[it->first] = it->second;
      temp_buffers.push_back(it->second);
      indices.emplace(it->first, indices.size());
    }

    // Finalize argument indices for each function.
    FuncArgumentIndices arg_indices;
    for (const auto &arg : op_context.input_args_) {
      arg_indices.inputs.push_back(indices.at(arg));
    }
    for (const auto &arg : op_context.output_args_) {
      arg_indices.outputs.push_back(indices.at(arg));
    }
    func_args.push_back(std::move(arg_indices));
  });

  return std::make_shared<TestExecutable>(this->name_, entry_params.size(),
                                          return_params, temp_buffers,
                                          func_codes, func_args);
}

}  // namespace xrt
}  // namespace oneflow
