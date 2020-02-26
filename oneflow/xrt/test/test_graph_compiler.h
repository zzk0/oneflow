#ifndef ONEFLOW_XRT_TEST_TEST_GRAPH_COMPILER_H_
#define ONEFLOW_XRT_TEST_TEST_GRAPH_COMPILER_H_

#include "oneflow/xrt/graph_compiler.h"

namespace oneflow {
namespace xrt {

class TestGraphCompiler : public GraphCompiler::Impl {
 public:
  explicit TestGraphCompiler(const std::string &name)
      : GraphCompiler::Impl(name) {}

  virtual ~TestGraphCompiler() = default;

  std::shared_ptr<Executable> Compile(
      const XrtGraph *graph,
      const std::vector<Parameter> &entry_params,
      const std::vector<Parameter> &return_params,
      const std::vector<InputOutputAlias> &aliases) override;
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TEST_TEST_GRAPH_COMPILER_H_
