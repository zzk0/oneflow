#include "oneflow/core/operator/prelu_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void PReluOp::InitFromOpConf() {
  CHECK(op_conf().has_prelu_conf());
  StrFieldTolower("data_format");
  EnrollInputBn("in");
  if (GetValFromCustomizedConf<std::string>("alpha").empty()) {
    EnrollTmpBn("alpha");
  } else {
    EnrollInputBn("alpha");
  }
  EnrollOutputBn("out")->set_mutable_inplace_ibn("in");
}

const PbMessage& PReluOp::GetCustomizedConf() const { return op_conf().prelu_conf(); }

void PReluOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  const PReluOpConf& conf = op_conf().prelu_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  BlobDesc* alpha_blob_desc = GetBlobDesc4BnInOp("alpha");
  if (conf.channel_shared()) {
    alpha_blob_desc->mut_shape() = Shape({1});
  } else {
    if (conf.data_format() == "channels_first") {
      alpha_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(1)});
    } else if (conf.data_format() == "channels_last") {
      alpha_blob_desc->mut_shape() =
          Shape({in_blob_desc->shape().At(in_blob_desc->shape().NumAxes() - 1)});
    } else {
      UNIMPLEMENTED();
    }
  }
  alpha_blob_desc->set_data_type(in_blob_desc->data_type());
}

void PReluOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = *HasBatchDim4BnInOp("in");
}

void PReluOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(output_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kPreluConf, PReluOp);

}  // namespace oneflow
