#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class PadGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PadGradOp);
  PadGradOp() = default;
  ~PadGradOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_pad_grad_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().pad_grad_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
    BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
    out_blob->set_data_type(in_blob->data_type());
    const int32_t padding_w =
        op_conf().pad_grad_conf().pad_left() + op_conf().pad_grad_conf().pad_right();
    const int32_t padding_h =
        op_conf().pad_grad_conf().pad_top() + op_conf().pad_grad_conf().pad_bottom();
    out_blob->mut_shape() =
        Shape({in_blob->shape().At(0), in_blob->shape().At(1), in_blob->shape().At(2) - padding_h,
               in_blob->shape().At(3) - padding_w});
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
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .MakeSplitSignatureListBuilder(
            JUST(LogicalBlobDesc4Ibn(input_bns().Get(0)))->shape().NumAxes())
        .Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kPadGradConf, PadGradOp);

}  // namespace oneflow
