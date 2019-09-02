#include "oneflow/core/operator/where_grad_op.h"

namespace oneflow {

void WhereGradOp::InitFromOpConf() {
  CHECK(op_conf().has_where_grad_conf());
  EnrollInputBn("condition", false);
  EnrollInputBn("out_diff");
  EnrollOutputBn("x_diff");
  EnrollOutputBn("y_diff");
}

const PbMessage& WhereGradOp::GetCustomizedConf() const { return op_conf().where_conf(); }

Maybe<void> WhereGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("x_diff") = *GetBlobDesc4BnInOp("out_diff");
  *GetBlobDesc4BnInOp("y_diff") = *GetBlobDesc4BnInOp("out_diff");
  return Maybe<void>::Ok();
}

void WhereGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(input_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kWhereGradConf, WhereGradOp);

}  // namespace oneflow
