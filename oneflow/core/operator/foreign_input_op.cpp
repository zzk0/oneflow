/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/operator/foreign_input_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/interface_op_util.h"

namespace oneflow {

namespace {

void CheckOpConf(const OperatorConf& op_conf) { CHECK(op_conf.ctrl_in_op_name().empty()); }

Maybe<void> InferBlobDescs(const OperatorConf& op_conf,
                           const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  CheckOpConf(op_conf);
  const auto& conf = op_conf.foreign_input_conf().blob_conf();
  BlobDesc* out_blob_desc = BlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape(conf.shape());
  CHECK_OR_RETURN(conf.has_data_type());
  out_blob_desc->set_data_type(conf.data_type());
  out_blob_desc->set_is_dynamic(conf.is_dynamic());
  return Maybe<void>::Ok();
}

}  // namespace

void ForeignInputOp::InitFromOpConf() {
  CHECK(op_conf().has_foreign_input_conf());
  if (op_conf().foreign_input_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", false);
}

Maybe<void> ForeignInputOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  CHECK_EQ_OR_RETURN(parallel_desc.parallel_num(), 1);
  return InferBlobDescs(op_conf(), BlobDesc4BnInOp);
}

Maybe<void> ForeignInputOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), 1);
  return InferBlobDescs(op_conf(), GetBlobDesc4BnInOp);
}

Maybe<void> ForeignInputOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  return Maybe<void>::Ok();
}

Maybe<void> ForeignInputOp::InferParallelDistributionSignature(
    ParallelDistributionSignature* parallel_distribution_signature,
    const ParallelDistributionSignature& parallel_distribution_constraints,
    const ParallelDesc& parallel_desc,
    std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
        ParallelDistributionInferHint4Ibn) const {
  const auto& parallel_hierarchy = parallel_desc.hierarchy();
  const InterfaceBlobConf& blob_conf = op_conf().foreign_input_conf().blob_conf();
  LOG(INFO) << "ForeignInputOp blob_conf" << blob_conf.DebugString();

  ParallelDistribution& in_parallel_distribution =
      (*parallel_distribution_signature->mutable_bn_in_op2parallel_distribution())["tick"];
  in_parallel_distribution.clear_sbp_parallel();
  FOR_RANGE(int64_t, i, 0, parallel_hierarchy->NumAxes()) {
    in_parallel_distribution.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
  }
  ParallelDistribution& out_parallel_distribution =
      (*parallel_distribution_signature->mutable_bn_in_op2parallel_distribution())["out"];
  InterfaceOpUtil::ParseParallelDistributionFromBlobConf(blob_conf, parallel_desc,
                                                         &out_parallel_distribution);
  LOG(INFO) << "ForeignInputOp op InferParallelDistributionSignature in:\n"
            << in_parallel_distribution.DebugString() << "\nout:\n"
            << out_parallel_distribution.DebugString();
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kForeignInputConf, ForeignInputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kForeignInputConf, 1);

}  // namespace oneflow
