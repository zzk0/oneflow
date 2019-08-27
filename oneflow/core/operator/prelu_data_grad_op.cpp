#include "oneflow/core/operator/prelu_data_grad_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void PReluDataGradOp::InitFromOpConf() {
  CHECK(op_conf().has_prelu_data_grad_conf());
  EnrollInputBn("dy", false);
  EnrollInputBn("alpha", false);
  EnrollInputBn("x", false);
  EnrollOutputBn("dx", false);
}

const PbMessage& PReluDataGradOp::GetCustomizedConf() const { return op_conf().prelu_data_grad_conf(); }

void PReluDataGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  const PReluDataGradOpConf& conf = this->op_conf().prelu_data_grad_conf();
  *GetBlobDesc4BnInOp("dx") = *GetBlobDesc4BnInOp("x");
}

void PReluDataGradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const PReluDataGradOpConf& conf = op_conf().prelu_data_grad_conf();
  //PbRf<int32_t>* perm = kernel_conf->mutable_prelu_data_grad_conf()->mutable_perm();
  PbRf<int32_t>* perm = kernel_conf->mutable_prelu_conf()->mutable_perm();
  const BlobDesc* x = GetBlobDesc4BnInOp("x");
  int64_t num_axes = x->shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, num_axes) { perm->Add(i); }
  if (!conf.channel_shared()) {
    if (conf.data_format() == "channels_first") {
      (*perm)[0] = 1;
      (*perm)[1] = 0;
    } else if (conf.data_format() == "channels_last") {
      (*perm)[num_axes - 1] = 0;
      (*perm)[0] = num_axes - 1;
    } else {
      UNIMPLEMENTED();
    }
  }  
}

void PReluDataGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(output_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kPreluDataGradConf, PReluDataGradOp);

}  // namespace oneflow
