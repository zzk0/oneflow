#include "oneflow/core/operator/broadcast_maximum_grad_op.h"

namespace oneflow {

void BroadcastMaximumGradOp::InitFromOpConf() {
  CHECK(op_conf().has_broadcast_maximum_grad_conf());
  EnrollInputBn("a");
  EnrollInputBn("b");
  EnrollInputBn("dy");
  EnrollOutputBn("da");
  EnrollOutputBn("db");
}

const PbMessage& BroadcastMaximumGradOp::GetCustomizedConf() const {
  return op_conf().broadcast_maximum_grad_conf();
}

Maybe<void> BroadcastMaximumGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  *GetBlobDesc4BnInOp("db") = *GetBlobDesc4BnInOp("b");
  *GetBlobDesc4BnInOp("da") = *GetBlobDesc4BnInOp("a");
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBroadcastMaximumGradConf, BroadcastMaximumGradOp);

}  // namespace oneflow