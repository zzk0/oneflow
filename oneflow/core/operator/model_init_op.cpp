#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ModelInitOp : public Operator {
 public:
  void InitFromOpConf() override;

  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void ModelInitOp::InitFromOpConf() {
  CHECK(op_conf().has_model_init_conf());
  EnrollInputBn("tick", false);
  EnrollRepeatedOutputBn("out", false);
  EnrollTmpBn("square_x_sum");
}

const PbMessage& ModelInitOp::GetCustomizedConf() const { return op_conf().model_init_conf(); }

Maybe<void> ModelInitOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const int64_t num_out = op_conf().model_init_conf().out().size();
  int64_t max_model_cnt = 0;
  int64_t square_x_sum_cnt = 0;
  FOR_RANGE(int64_t, i, 0, num_out) {
    const VariableOpConf& original_variable_conf =
        op_conf().model_init_conf().original_variable_conf(i);
    BlobDesc* out_i = GetBlobDesc4BnInOp(GenRepeatedBn("out", i));
    out_i->mut_shape() = Shape(original_variable_conf.shape());
    out_i->set_data_type(original_variable_conf.data_type());
    if (original_variable_conf.has_normalize_conf()) {
      square_x_sum_cnt = out_i->shape().elem_cnt()
                         / out_i->shape().At(original_variable_conf.normalize_conf().axis());
      if (square_x_sum_cnt > max_model_cnt) { max_model_cnt = square_x_sum_cnt; }
    }
  }
  *GetBlobDesc4BnInOp("square_x_sum") = *GetBlobDesc4BnInOp(GenRepeatedBn("out", 0));
  GetBlobDesc4BnInOp("square_x_sum")->mut_shape() = Shape({max_model_cnt});

  return Maybe<void>::Ok();
}

Maybe<void> ModelInitOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const std::string& bns : output_bns()) { BatchAxis4BnInOp(bns)->clear_value(); }
  return Maybe<void>::Ok();
}

Maybe<void> ModelInitOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(
          JUST(LogicalBlobDesc4Ibn(output_bns().Get(0)))->shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kModelInitConf, ModelInitOp);

}  // namespace oneflow
