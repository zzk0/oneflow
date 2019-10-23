#include "oneflow/core/operator/additive_angular_margin_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void AdditiveAngularMarginGradOp::InitFromOpConf() {
  CHECK(op_conf().has_additive_angular_margin_grad_conf());
  EnrollInputBn("dy");
  EnrollInputBn("label", false);
  EnrollInputBn("sin_theta_data", false);
  EnrollOutputBn("dx");
}

const PbMessage& AdditiveAngularMarginGradOp::GetCustomizedConf() const {
  return op_conf().additive_angular_margin_grad_conf();
}

Maybe<void> AdditiveAngularMarginGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
  CHECK_GT_OR_RETURN(dy->shape().NumAxes(), 0);
  const BlobDesc* label = GetBlobDesc4BnInOp("label");
  CHECK_GT_OR_RETURN(label->shape().NumAxes(), 0);
  CHECK_OR_RETURN(IsIntegralDataType(label->data_type()));
  CHECK_EQ_OR_RETURN(label->shape().At(0), dy->shape().At(0));

  const BlobDesc* sin_theta_data = GetBlobDesc4BnInOp("sin_theta_data");
  CHECK_EQ_OR_RETURN(sin_theta_data->shape().At(0), label->shape().At(0));

  *GetBlobDesc4BnInOp("dx") = *dy;
  return Maybe<void>::Ok();
}

Maybe<void> AdditiveAngularMarginGradOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("label", 0)
      .Split("dy", 0)
      .Split("sin_theta_data", 0)
      .Split("dx", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder()
      .Broadcast("label")
      .Broadcast("sin_theta_data")
      .Split("dy", 1)
      .Split("dx", 1)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());

  return Maybe<void>::Ok();
}

Maybe<void> AdditiveAngularMarginGradOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  *BatchAxis4BnInOp("dx") = *BatchAxis4BnInOp("dy");
  return Maybe<void>::Ok();
}

void AdditiveAngularMarginGradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx,
    std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp) const {
  const int64_t dim = op_conf().additive_angular_margin_grad_conf().depth();
  if (dim > 0) {
    CHECK_GE(dim, parallel_ctx->parallel_num());
    BalancedSplitter bs(dim, parallel_ctx->parallel_num());
    kernel_conf->mutable_additive_angular_margin_grad_conf()->set_lower_bound(
        bs.At(parallel_ctx->parallel_id()).begin());
  } else {
    kernel_conf->mutable_additive_angular_margin_grad_conf()->set_lower_bound(0);
  }
}

REGISTER_OP(OperatorConf::kAdditiveAngularMarginGradConf, AdditiveAngularMarginGradOp);

}  // namespace oneflow
