#include "oneflow/core/operator/leaky_relu_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void LeakyReluOp::InitFromOpConf() {
  CHECK(op_conf().has_leaky_relu_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("in");
}

const PbMessage& LeakyReluOp::GetCustomizedConf() const { return op_conf().leaky_relu_conf(); }

Maybe<void> LeakyReluOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

Maybe<void> LeakyReluOp::GetSbpSignatures(
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

REGISTER_OP(OperatorConf::kLeakyReluConf, LeakyReluOp);

}  // namespace oneflow