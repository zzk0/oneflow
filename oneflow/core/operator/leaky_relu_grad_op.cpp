#include "oneflow/core/operator/leaky_relu_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void LeakyReluGradOp::InitFromOpConf() {
  CHECK(op_conf().has_leaky_relu_grad_conf());
  EnrollInputBn("x");
  EnrollInputBn("dy");
  EnrollOutputBn("dx")->set_mutable_inplace_ibn("dy");
}

const PbMessage& LeakyReluGradOp::GetCustomizedConf() const { return op_conf().leaky_relu_grad_conf(); }

Maybe<void> LeakyReluGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("dx") = *GetBlobDesc4BnInOp("x");
  return Maybe<void>::Ok();
}

Maybe<void> LeakyReluGradOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(
          JUST(LogicalBlobDesc4Ibn(input_bns().Get(0)))->shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kLeakyReluGradConf, LeakyReluGradOp);

}  // namespace oneflow