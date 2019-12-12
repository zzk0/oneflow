#include "oneflow/core/operator/operator.h"

namespace oneflow {

class BatchCalcIoUMatrixOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchCalcIoUMatrixOp);
  BatchCalcIoUMatrixOp() = default;
  ~BatchCalcIoUMatrixOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_batch_calc_iou_matrix_conf());
    EnrollInputBn("batch_boxes1", false);
    EnrollInputBn("boxes2", false);
    EnrollOutputBn("iou_matrix", false);
  }
  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().batch_calc_iou_matrix_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    // input: boxes1 (B, M, 4)
    const BlobDesc* batch_boxes1 = GetBlobDesc4BnInOp("batch_boxes1");
    CHECK_EQ_OR_RETURN(batch_boxes1->shape().NumAxes(), 3);
    CHECK_EQ_OR_RETURN(batch_boxes1->shape().At(2), 4);
    // input: boxes2 (G, 4)
    const BlobDesc* boxes2 = GetBlobDesc4BnInOp("boxes2");
    CHECK_EQ_OR_RETURN(boxes2->shape().NumAxes(), 2);
    CHECK_EQ_OR_RETURN(boxes2->shape().At(1), 4);
    const int32_t num_boxes2 = boxes2->shape().At(0);
    // output: iou_matrix (B, M, G)
    BlobDesc* iou_matrix = GetBlobDesc4BnInOp("iou_matrix");
    iou_matrix->mut_shape() =
        Shape({batch_boxes1->shape().At(0), batch_boxes1->shape().At(1), num_boxes2});
    iou_matrix->set_data_type(DataType::kFloat);
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("batch_boxes1", 0)
        .Broadcast("boxes2")
        .Split("iou_matrix", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kBatchCalcIouMatrixConf, BatchCalcIoUMatrixOp);

}  // namespace oneflow
