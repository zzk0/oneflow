#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_split_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf op_conf;
    op_conf.set_name(op.op_name() + "_grad");
    ConcatOpConf* conf = op_conf.mutable_concat_conf();
    for (const auto& obn : op.output_bns()) {
      if (DiffLbi4BnInOp(obn) != nullptr) {
        *conf->mutable_in()->Add() = GenLogicalBlobName(*DiffLbi4BnInOp(obn));
      }
    }
    conf->set_axis(op.op_conf().split_conf().axis());
    conf->set_out("out");
    op_confs->push_back(op_conf);
    DiffLbi4BnInOp("in")->set_op_name(op_conf.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSplitConf, &GenerateBackwardOpConf);

}  // namespace oneflow
