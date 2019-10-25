#include "oneflow/core/operator/sparse_cross_entropy_ms1_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void SparseCrossEntropyMs1GradOp::InitFromOpConf() {
  CHECK(op_conf().has_sparse_cross_entropy_ms1_grad_conf());
  EnrollInputBn("prediction", false);
  EnrollInputBn("label");
  EnrollInputBn("dy");
  EnrollOutputBn("prediction_diff");
}

const PbMessage& SparseCrossEntropyMs1GradOp::GetCustomizedConf() const {
  return op_conf().sparse_cross_entropy_ms1_grad_conf();
}

Maybe<void> SparseCrossEntropyMs1GradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  *GetBlobDesc4BnInOp("prediction_diff") = *GetBlobDesc4BnInOp("prediction");
  return Maybe<void>::Ok();
}

Maybe<void> SparseCrossEntropyMs1GradOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("prediction", 1)
      .Broadcast("dy")
      .Broadcast("label")
      .Split("prediction_diff", 1)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

void SparseCrossEntropyMs1GradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const int64_t dim = op_conf().sparse_cross_entropy_ms1_grad_conf().depth();
  CHECK_GE(dim, parallel_ctx->parallel_num());
  BalancedSplitter bs(dim, parallel_ctx->parallel_num());
  kernel_conf->mutable_sparse_cross_entropy_grad_conf()->set_lower_bound(
      bs.At(parallel_ctx->parallel_id()).begin());
}

REGISTER_OP(OperatorConf::kSparseCrossEntropyMs1GradConf, SparseCrossEntropyMs1GradOp);

}  // namespace oneflow
