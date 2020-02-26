#include "oneflow/xrt/test/test_op_kernel.h"
#include "oneflow/xrt/parameter.h"

namespace oneflow {
namespace xrt {

// Computing relu implemented in test_relu_op.cu.
// TODO(hjchen2): Support compute stream.
extern void ComputeRelu(/*void *stream, */const Parameter &input,
                        const Parameter &output);

Parameter CreateParameter(const std::string &name, const Shape &shape,
                          const DataType &data_type) {
  // TODO(hjchen2)
  Parameter param;
  return param;
}

class TestReluOpKernel : public TestOpKernel {
 public:
  void Compile(TestOpContext *ctx) override {
    ctx->func_code_ = [](const std::vector<Parameter> &inputs,
                         const std::vector<Parameter> &outputs) {
      CHECK_EQ(inputs.size(), 1);
      CHECK_EQ(outputs.size(), 1);
      ComputeRelu(inputs[0], outputs[0]);
    };

    for (const XrtEdge *edge : ctx->node_->in_edges()) {
      const auto &name = edge->argument().name();
      CHECK_GT(ctx->all_params_.count(name), 0);
      // TODO(hjchen2): Filter duplicate input.
      ctx->input_args_.push_back(name);
    }

    for (const XrtEdge *edge : ctx->node_->out_edges()) {
      const auto &name = edge->argument().name();
      // TODO(hjchen2): Filter duplicate output.
      ctx->output_args_.push_back(name);
      if (ctx->all_params_.count(name) == 0 &&
          ctx->tmp_buffers_.count(name) == 0) {
        auto param = CreateParameter(name /*argument name*/,
                                     edge->argument().shape(),
                                     edge->argument().data_type());
        ctx->tmp_buffers_[name] = std::move(param);
      }
    }
  }
};

REGISTER_TEST_OP_KERNEL(Relu, TestReluOpKernel).EnableTrainPhase().Finalize();

}  // namespace xrt
}  // namespace oneflow
