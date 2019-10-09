#ifndef ONEFLOW_CORE_OPERATOR_ARC_FACE_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_ARC_FACE_GRAD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ArcFaceGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArcFaceGradOp);
  ArcFaceGradOp() = default;
  ~ArcFaceGradOp() override = default;
  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;

  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx,
      std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp) const override;

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ARC_FACE_GRAD_OP_H_
