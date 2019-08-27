#include "oneflow/core/operator/prelu_alpha_grad_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void PReluAlphaGradOp::InitFromOpConf() {
  CHECK(op_conf().has_prelu_alpha_grad_conf());
  EnrollInputBn("dy", false);
  EnrollInputBn("x", false);
  EnrollOutputBn("alpha_diff", false);
}

const PbMessage& PReluAlphaGradOp::GetCustomizedConf() const { return op_conf().prelu_alpha_grad_conf(); }

void PReluAlphaGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  const PReluAlphaGradOpConf& conf = op_conf().prelu_alpha_grad_conf();
  const BlobDesc* x = GetBlobDesc4BnInOp("x");
  BlobDesc* alpha_diff_blob_desc = GetBlobDesc4BnInOp("alpha_diff");
  if (conf.channel_shared()) {
    alpha_diff_blob_desc->mut_shape() = Shape({1});
  } else {
    if (conf.data_format() == "channels_first") {
      alpha_diff_blob_desc->mut_shape() = Shape({x->shape().At(1)});
    } else if (conf.data_format() == "channels_last") {
      alpha_diff_blob_desc->mut_shape() =
          Shape({x->shape().At(x->shape().NumAxes() - 1)});
    } else {
      UNIMPLEMENTED();
    }
  }
  alpha_diff_blob_desc->set_data_type(x->data_type());
}

void PReluAlphaGradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const PReluAlphaGradOpConf& conf = op_conf().prelu_alpha_grad_conf();
  //PbRf<int32_t>* perm = kernel_conf->mutable_prelu_alpha_grad_conf()->mutable_perm();
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

void PReluAlphaGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(output_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kPreluAlphaGradConf, PReluAlphaGradOp);

}  // namespace oneflow
