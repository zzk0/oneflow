#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/register/dense_shape_view.h"

namespace oneflow {

class DeviceReduceSumOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceReduceSumOp);
  DeviceReduceSumOp() = default;
  ~DeviceReduceSumOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_device_reduce_sum_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out");
    if (op_conf().device_reduce_sum_conf().has_in_sys()) {
      EnrollTmpBn("fw_tmp");
    } else {
      EnrollTmpBn("fw_tmp");
    }
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().device_reduce_sum_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
    *GetBlobDesc4BnInOp("fw_tmp") = *in_blob;
    BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
    out_blob->set_data_type(in_blob->data_type());
    out_blob->mut_shape() = {parallel_ctx->parallel_num()};
    LOG(INFO)<<"out_blob->mut_shape()"<<parallel_ctx->parallel_num();
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
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kDeviceReduceSumConf, DeviceReduceSumOp);

}  // namespace oneflow
