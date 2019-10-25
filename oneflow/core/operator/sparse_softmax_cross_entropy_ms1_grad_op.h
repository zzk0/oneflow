#ifndef ONEFLOW_CORE_OPERATOR_SPARSE_SOFTMAX_CROSS_ENTROPY_MS1_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_SPARSE_SOFTMAX_CROSS_ENTROPY_MS1_GRAD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SparseSoftmaxCrossEntropyMs1GradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseSoftmaxCrossEntropyMs1GradOp);
  SparseSoftmaxCrossEntropyMs1GradOp() = default;
  ~SparseSoftmaxCrossEntropyMs1GradOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SPARSE_SOFTMAX_CROSS_ENTROPY_MS1_GRAD_OP_H_
