#ifndef ONEFLOW_CORE_OPERATOR_ADDITIVE_ANGULAR_MARGIN_OP_H_
#define ONEFLOW_CORE_OPERATOR_ADDITIVE_ANGULAR_MARGIN_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AdditiveAngularMarginOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AdditiveAngularMarginOp);
  AdditiveAngularMarginOp() = default;
  ~AdditiveAngularMarginOp() override = default;
  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx,
                             const SbpSignature* sbp_signature) const override;

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ADDITIVE_ANGULAR_MARGIN_OP_H_
