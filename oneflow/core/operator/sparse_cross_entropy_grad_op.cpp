#include "oneflow/core/operator/sparse_cross_entropy_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void SparseCrossEntropyGradOp::InitFromOpConf() {
  CHECK(op_conf().has_sparse_cross_entropy_grad_conf());
  EnrollInputBn("prediction", false);
  EnrollInputBn("label");
  EnrollInputBn("dy");
  EnrollOutputBn("prediction_diff");
}

const PbMessage& SparseCrossEntropyGradOp::GetCustomizedConf() const {
  return op_conf().sparse_cross_entropy_grad_conf();
}

Maybe<void> SparseCrossEntropyGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  *GetBlobDesc4BnInOp("prediction_diff") = *GetBlobDesc4BnInOp("prediction");
  return Maybe<void>::Ok();
}

Maybe<void> SparseCrossEntropyGradOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder()
      .Split("prediction", 1)
      .Broadcast("dy")
      .Broadcast("label")
      .Split("prediction_diff", 1)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

void SparseCrossEntropyGradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const int64_t dim = op_conf().sparse_cross_entropy_grad_conf().depth();
  if (dim > 0) {
    CHECK_GE(dim, parallel_ctx->parallel_num());
    BalancedSplitter bs(dim, parallel_ctx->parallel_num());
    kernel_conf->mutable_sparse_cross_entropy_grad_conf()->set_lower_bound(
        bs.At(parallel_ctx->parallel_id()).begin());
  } else {
    kernel_conf->mutable_sparse_cross_entropy_grad_conf()->set_lower_bound(0);
  }
}

REGISTER_OP(OperatorConf::kSparseCrossEntropyGradConf, SparseCrossEntropyGradOp);

}  // namespace oneflow
